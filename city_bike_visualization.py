from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource
from bokeh.models import Select, LinearColorMapper, ColorBar, LassoSelectTool
from bokeh.models.widgets import MultiSelect, DateRangeSlider, RangeSlider, RadioButtonGroup
from bokeh.palettes import Category20
from bokeh.plotting import figure, curdoc
from bokeh.core.properties import value
from bokeh.tile_providers import CARTODBPOSITRON
from bokeh.transform import transform

import datetime
import json
import numpy as np
import pandas as pd
import pickle
import pyproj
from time import time
import os.path  # getmtime, isfile
import urllib.request


class CityBikeAnalysis:
    def __init__(self, log_file, city='Oslo', coord_mapping_file='', altitude_file=''):

        self.city = city
        self.station_info_file = 'station_info.csv'
        self.df_s_info = pd.read_csv(self.station_info_file, index_col=0)
        self.df_s_info.index = self.df_s_info.index.astype(str)

        self.tz_offset = pd.Timedelta('0h')

        self.log_file = log_file
        self.df_log = pd.read_csv(log_file)
        self.re_index()
        self.stations = [str(s_id) for s_id in np.sort(self.df_log['station_id'].unique())]

        self.df_bikes_avail = pd.DataFrame()
        self.df_docs_avail = pd.DataFrame()
        self.pivot_df()

        self.df_bikes_source = pd.DataFrame()
        self.df_docks_source = pd.DataFrame()

        if os.path.isfile(altitude_file):
            self.df_altitude = pd.read_csv(altitude_file, index_col=0)

        self.id_coord_mapping_df = pd.DataFrame()

        if os.path.isfile(coord_mapping_file):
            s_mod = os.path.getmtime(coord_mapping_file)
            s_now = datetime.datetime.now().timestamp()
            if s_mod < (s_now - 3600):
                self.id_coord_mapping_df = self.make_station_coord_transformed_df(self.get_station_info())
                with open(coord_mapping_file, 'wb') as handle:
                    pickle.dump(self.id_coord_mapping_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    # TODO Do not overwrite?
            else:
                with open('mapping_df.pickle', 'rb') as handle:
                    self.id_coord_mapping_df = pickle.load(handle)

        if city == 'Oslo':
            self.mid_lat, self.mid_lon = pyproj.transform(pyproj.Proj(init='epsg:4326'),
                                                          pyproj.Proj(init='epsg:3857'),
                                                          10.735, 59.928)  # Center of map location
            self.coord_range_delta = 7500
        elif city == 'NewYorkCity':
            self.mid_lat, self.mid_lon = pyproj.transform(pyproj.Proj(init='epsg:4326'),
                                                          pyproj.Proj(init='epsg:3857'),
                                                          -73.973, 40.736)
            self.coord_range_delta = 12500

        # Colors
        self.empty_color = "#7e7e7e"
        self.avail_color = "#87e03e"
        self.full_color = "#ff2332"

        x_mul_sel_opt = [(s, s + ' | ' + name) for s, name in
                         zip(self.stations, self.df_s_info.reindex(self.stations, fill_value='Unknown')['name'])]
        self.x_mul_select = MultiSelect(title="Stations", value=[self.stations[0], self.stations[1]],
                                        options=x_mul_sel_opt, size=8)
        self.x_mul_select.on_change('value', self.x_mul_select_callback)

        s_date = datetime.date.fromtimestamp(self.df_bikes_avail.index[0].value/10**9)
        e_date = datetime.date.fromtimestamp(self.df_bikes_avail.index[-1].value/10**9)
        e_date += datetime.timedelta(days=1)
        self.dr_slider = DateRangeSlider(title='Date Range', start=s_date, end=e_date, step=24,
                                         value=(e_date - datetime.timedelta(days=5), e_date),
                                         callback_throttle=500)
        self.dr_slider.on_change('value', self.date_rage_slider_callback)

        self.range_slider = RangeSlider(title='Time of day', start=5, end=25, step=1, value=(5, 25))
        self.range_slider.on_change('value', self.hour_rage_slider_callback)

        self.histogram_policy = RadioButtonGroup(labels=["Aggregate", "Flatten"], active=0)
        self.histogram_policy.on_change('active', self.histogram_policy_callback)

        self.x_select = Select(title="X-axis", value=self.stations[0], options=self.stations)
        self.x_select.on_change('value', self.x_select_callback)
        self.y_select = Select(title="Y-axis", value=self.stations[1], options=self.stations)
        self.y_select.on_change('value', self.y_select_callback)

        self.create_sliced_df()
        self.station_history_fig, self.station_history_fig_source = self.create_station_history_fig()
        self.station_avail_histogram, self.station_avail_histogram_source = self.create_station_avail_histogram()
        self.extremes_annulus, self.extremes_annulus_source = self.create_extremes_figure()
        self.station_map_fig, self.station_map_fig_source = self.create_station_map()
        self.station_map_fig_source.selected.indices = [self.id_coord_mapping_df.index.get_loc(s) for s in
                                                        self.x_mul_select.value]

        self._layout = column(row(column(self.x_mul_select,
                                         self.dr_slider,
                                         self.range_slider,
                                         self.histogram_policy,
                                         self.extremes_annulus),
                                  self.station_avail_histogram,
                                  self.station_map_fig),
                              row(self.station_history_fig))
                              # row(column(self.x_select, self.y_select), self.two_station_heatmap()))

    @property
    def layout(self):
        return self._layout

    def re_index(self):
        self.df_log.index = pd.to_datetime(10**9*self.df_log['last_reported'])
        self.df_log.drop(['last_reported'], axis=1, inplace=True)
        self.df_log.index.tz_localize('UTC').tz_convert('Europe/Oslo')

    def pivot_df(self):

        def get_column_mapper(df):
            d = {}
            for col in df.columns.values:
                d[col] = str(col)
            return d

        self.df_bikes_avail = self.df_log.pivot_table(index='last_reported',
                                                      columns='station_id',
                                                      values='num_bikes_available')
        self.df_bikes_avail.rename(columns=get_column_mapper(self.df_bikes_avail), inplace=True)

        self.df_docs_avail = self.df_log.pivot_table(index='last_reported',
                                                     columns='station_id',
                                                     values='num_docks_available')
        self.df_docs_avail.rename(columns=get_column_mapper(self.df_docs_avail), inplace=True)

    def create_sliced_df(self):
        s_hour = self.range_slider.value[0]
        e_hour = self.range_slider.value[1]
        if e_hour == 25:
            hours_of_day = self.df_bikes_avail.index.hour.isin(np.append([0], range(s_hour, e_hour - 1)))
        else:
            hours_of_day = self.df_bikes_avail.index.hour.isin(range(s_hour, e_hour))

        df_bikes_sliced = self.df_bikes_avail.loc[hours_of_day][
                          self.dr_slider.value_as_datetime[0]:self.dr_slider.value_as_datetime[1]]
        df_docks_sliced = self.df_docs_avail.loc[hours_of_day][
                          self.dr_slider.value_as_datetime[0]:self.dr_slider.value_as_datetime[1]]

        self.df_bikes_source = df_bikes_sliced.loc[:, [val for val in self.x_mul_select.value]].fillna(0)   # TODO  Filling nan with 0 might not be best solution
        self.df_docks_source = df_docks_sliced.loc[:, [val for val in self.x_mul_select.value]].fillna(0)  #

    def create_station_history_fig(self):

        source_dict = self.create_station_history_fig_source_dict()
        source = ColumnDataSource(source_dict)

        fig = figure(plot_width=800, plot_height=600,
                     x_axis_type='datetime')

        colors = Category20[20][::2]
        for val, color in zip(self.x_mul_select.value, colors):
            fig.step('last_reported', val,
                     color=color, legend=value(val), source=source)

        return fig, source

    def create_station_history_fig_source_dict(self):
        df_sliced = self.df_bikes_avail[self.dr_slider.value_as_datetime[0]:self.dr_slider.value_as_datetime[1]]
        df_source = df_sliced.loc[:, [val for val in self.x_mul_select.value]]
        df_source.index = df_source.index + self.tz_offset
        df_source = df_source.reset_index()
        return df_source.to_dict(orient='list')

    def create_station_avail_histogram(self):

        d = self.create_station_avail_histogram_source_dict()
        source = ColumnDataSource(d)

        fig = figure(plot_width=800, plot_height=600,
                     title='Histogram of availability', background_fill_color="#fafafa")

        fig.quad(top='top', bottom=0, left='left', right='right',
                 color="colors", line_color="white", alpha=0.9,
                 source=source)

        fig.y_range.start = 0
        # fig.legend.location = "center_right"
        # fig.legend.background_fill_color = "#fefefe"
        x_labl = 'x, Available bikes in station' if len(self.x_mul_select.value) == 1 else 'x, Available bikes in stations'
        fig.xaxis.axis_label = x_labl
        fig.yaxis.axis_label = 'P(x)'
        fig.grid.grid_line_color = "white"

        return fig, source

    def create_station_avail_histogram_source_dict(self):

        if self.histogram_policy.labels[self.histogram_policy.active] == 'Flatten':
            df_bikes_source = self.df_bikes_source
            df_docks_source = self.df_docks_source
        elif self.histogram_policy.labels[self.histogram_policy.active] == 'Aggregate':
            df_bikes_source = self.df_bikes_source.sum(axis=1)
            df_docks_source = self.df_docks_source.sum(axis=1)
        else:
            df_bikes_source = self.df_bikes_source
            df_docks_source = self.df_docks_source

        b = np.array((range(-1, int(df_bikes_source.max().max()) + 1))) + 0.5
        hist, edges = np.histogram(df_bikes_source, density=True, bins=b)

        colors = np.array([self.avail_color] * (len(hist)))

        if np.any(df_bikes_source.values == 0):
            colors[0] = self.empty_color

        if np.any(df_docks_source.values == 0):
            colors[-1] = self.full_color  # TODO Add for all

        d = dict(top=hist, left=edges[:-1], right=edges[1:], colors=colors)

        return d

    def create_extremes_figure(self):

        fig = figure(title='Availability summary',
                     width=300, height=300,
                     x_range=(-100, 100), y_range=(-100, 100),
                     tools='')
        fig.axis.visible = False
        fig.grid.visible = False
        fig.outline_line_alpha = 0.0

        source = ColumnDataSource(self.create_extremes_figure_source_dict())

        fig.annular_wedge(x=0, y=0,
                          inner_radius=50, outer_radius=75,
                          start_angle=0, end_angle='end_angle_empty',
                          color=self.empty_color,
                          source=source)  # Totally empty
        fig.annular_wedge(x=0, y=0,
                          inner_radius=50, outer_radius=75,
                          start_angle='end_angle_empty', end_angle='end_angle_mid',
                          color=self.avail_color,
                          source=source)  # Not empty, not full
        fig.annular_wedge(x=0, y=0,
                          inner_radius=50, outer_radius=75,
                          start_angle='end_angle_mid', end_angle=2*np.pi,
                          color=self.full_color,
                          source=source)  # Totally full

        fig.circle(x=0, y=0, radius=50, color="#fafafa")

        fig.text(x=0, y=14, text='empty_percent', text_color=self.empty_color, text_baseline='bottom',
                 text_align='center', text_font_size='16pt', text_font_style='bold', source=source)
        fig.text(x=0, y=0, text='avail_percent', text_color=self.avail_color, text_baseline='middle',
                 text_align='center', text_font_size='21pt', text_font_style='bold', source=source)
        fig.text(x=0, y=-14, text='full_percent', text_color=self.full_color, text_baseline='top',
                 text_align='center', text_font_size='16pt', text_font_style='bold', source=source)

        return fig, source

    def create_extremes_figure_source_dict(self):

        df_bikes_source = self.df_bikes_source.sum(axis=1)
        df_docks_source = self.df_docks_source.sum(axis=1)

        d = dict(end_angle_empty=[2*np.pi*np.count_nonzero(df_bikes_source.values == 0)/len(df_bikes_source.values)],
                 end_angle_mid=[2*np.pi*(1-(np.count_nonzero(df_docks_source.values == 0) / len(df_docks_source.values)))])

        d['empty_percent'] = [f"{100 * (np.abs(d['end_angle_empty'][0]) / (2 * np.pi)):2.1f}% "]  # TODO deal with nan
        d['avail_percent'] = [f"{100*(np.abs(d['end_angle_mid'][0] - d['end_angle_empty'][0])/(2 * np.pi)):.1f}%"] # TODO deal with nan
        d['full_percent'] = [f"{100 * (np.abs(2 * np.pi - d['end_angle_mid'][0]) / (2 * np.pi)):.1f}%"]  # TODO deal with nan

        return d

    def create_station_map(self):

        fig = figure(plot_width=780, plot_height=600, title='Stations map and selector',
                     tools=['pan', 'box_zoom', 'wheel_zoom', 'lasso_select', 'reset'],
                     x_range=(self.mid_lat - self.coord_range_delta, self.mid_lat + self.coord_range_delta),
                     y_range=(self.mid_lon - self.coord_range_delta, self.mid_lon + self.coord_range_delta),
                     x_axis_type="mercator", y_axis_type="mercator")
        fig.add_tile(CARTODBPOSITRON)

        lst = fig.select(dict(type=LassoSelectTool))[0]
        lst.select_every_mousemove = False

        # # Bikes available
        # fig.annular_wedge(x=status_df['x_proj'], y=status_df['y_proj'], color=self.avail_color,
        #                 inner_radius=np.zeros(len(status_df)), outer_radius=25 * np.sqrt(status_df['num_docs']),
        #                 start_angle=(np.pi / 2 + np.zeros(len(status_df))),
        #                 end_angle=(np.pi / 2 - status_df['docks_start_ang']), direction='clock')
        # # Docks available
        # fig.annular_wedge(x=status_df['x_proj'], y=status_df['y_proj'], color="#ea6d3f",
        #                 inner_radius=np.zeros(len(status_df)), outer_radius=25 * np.sqrt(status_df['num_docs']),
        #                 start_angle=(np.pi / 2 - status_df['docks_start_ang']),
        #                 end_angle=(10 ** (-3) + np.pi / 2 * np.ones(len(status_df))), direction='clock')
        #
        # fig.text(x=status_df['x_proj'], y=status_df['y_proj'], text=status_df.index)

        source = ColumnDataSource(self.id_coord_mapping_df)

        c = fig.circle(x='x_proj', y='y_proj', size=10, color="navy", source=source)
        fig.text(x='x_proj', y='y_proj', text='station_id', source=source)

        c.data_source.selected.on_change('indices', self.lasso_select_callback)

        return fig, source

    def two_station_heatmap(self):
        fig = figure(width=700,
                     match_aspect=True,
                     tools='')
        fig.xgrid.grid_line_color = None
        fig.ygrid.grid_line_color = None
        s_id1, s_id2 = self.x_select.value, self.y_select.value
        df_counts = self.df_bikes_avail.groupby([s_id1, s_id2]).size().reset_index(name='counts')
        df_counts.rename(columns={s_id1: s_id1, s_id2: s_id2}, inplace=True)
        source = ColumnDataSource(df_counts)

        pallette = ['#084594', '#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef', '#deebf7', '#f7fbff'][::-1]
        mapper = LinearColorMapper(palette=pallette, low=0, high=df_counts.counts.max())
        color_bar = ColorBar(color_mapper=mapper)

        fig.rect(x=str(s_id1), y=str(s_id2),
                 width=1.0, height=1.0,
                 line_alpha = 0.0,
                 fill_color=transform('counts', mapper), source=source)

        fig.add_layout(color_bar, 'right')

        return fig

    def lasso_select_callback(self, attr, old, new):
        # print(self.id_coord_mapping_df.iloc[new].index.values)
        if new:
            self.x_mul_select.value = list(self.id_coord_mapping_df.iloc[new].index.values)

    def get_station_info(self):

        if self.city == 'Oslo':
            json_url = "https://gbfs.urbansharing.com/oslobysykkel.no/station_information.json"
        elif self.city == 'NewYorkCity':
            json_url = "https://gbfs.citibikenyc.com/gbfs/es/station_information.json"

        with urllib.request.urlopen(json_url) as url:
            station_info = json.loads(url.read().decode())
            return station_info

    # @staticmethod
    def make_station_coord_transformed_df(self, station_info_in):

        def apply_transfomation(lon_in, lat_in):
            return pyproj.transform(pyproj.Proj(init='epsg:4326'), pyproj.Proj(init='epsg:3857'), lon_in, lat_in)

        lon = [s['lon'] for s in station_info_in['data']['stations']]
        lat = [s['lat'] for s in station_info_in['data']['stations']]
        x_proj, y_proj = zip(*map(apply_transfomation, lon, lat))
        index = [s['station_id'] for s in station_info_in['data']['stations']]
        df = pd.DataFrame(data={'x_proj': x_proj, 'y_proj': y_proj}, index=index)
        df.index.name = 'station_id'
        return df

    def update(self):
        t_start = time()
        self.station_history_fig, self.station_history_fig_source = self.create_station_history_fig()
        self.station_avail_histogram, self.station_avail_histogram_source = self.create_station_avail_histogram()
        print(f"Used: {(time() - t_start) * 1000} ms to regenerate figures")
        t_start = time()
        self._layout.children[0].children[1] = self.station_avail_histogram
        self._layout.children[1].children[0] = self.station_history_fig
        print(f"Used: {(time() - t_start)*1000} ms to update layout")
        # self._layout.children[1].children[1] = self.two_station_heatmap()

    def y_select_callback(self, attr, old, new):
        self.update()

    def x_select_callback(self, attr, old, new):
        self.update()

    def x_mul_select_callback(self, attr, old, new):
        self.create_sliced_df()
        self.station_history_fig, self.station_history_fig_source = self.create_station_history_fig()
        self.station_avail_histogram, self.station_avail_histogram_source = self.create_station_avail_histogram()
        self.extremes_annulus_source.data = self.create_extremes_figure_source_dict()
        self.station_map_fig_source.selected.indices = [self.id_coord_mapping_df.index.get_loc(s) for s in
                                                        self.x_mul_select.value]
        self._layout.children[0].children[1] = self.station_avail_histogram
        self._layout.children[1].children[0] = self.station_history_fig

    def date_rage_slider_callback(self, attr, old, new):
        t_start = time()
        self.create_sliced_df()
        self.station_history_fig_source.data = self.create_station_history_fig_source_dict()
        self.station_avail_histogram_source.data = self.create_station_avail_histogram_source_dict()
        self.extremes_annulus_source.data = self.create_extremes_figure_source_dict()
        print(f"Used: {(time() - t_start) * 1000} ms to calculate sources")

    def hour_rage_slider_callback(self, attr, old, new):
        t_start = time()
        # self.station_history_fig_source.data = self.create_station_history_fig_source_dict()
        self.create_sliced_df()
        self.extremes_annulus_source.data = self.create_extremes_figure_source_dict()
        self.station_avail_histogram_source.data = self.create_station_avail_histogram_source_dict()
        print(f"Used: {(time() - t_start) * 1000} ms to calculate sources")

    def histogram_policy_callback(self, attr, old, new):
        self.station_avail_histogram_source.data = self.create_station_avail_histogram_source_dict()


def main():
    log_file = 'live_datalog/OCB.csv'
    altitude = 'altitude.csv'
    coord_mapping = 'mapping_df.pickle'
    cbd = CityBikeAnalysis(log_file, 'Oslo', coord_mapping, altitude)
    curdoc().add_root(cbd.layout)
    curdoc().title = "OsloCityBike"


main()
