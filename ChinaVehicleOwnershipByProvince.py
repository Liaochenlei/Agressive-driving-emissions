import pandas as pd
from pyecharts.charts import Map
from pyecharts import options as opts

df = pd.read_excel('ChinaVehicleOwnership.xlsx', sheet_name='Sheet1')   # Read Data
vals = df.values

Year = 2018     # Enter all four digits of the year you want to display
Name = 'Car' + str(Year) + '.html'  # Generates filenames

province = []
data = []
namemap = {}


for i in vals[0:]:
    province.append(i[0].replace(' ', ''))  # Remove space
    namemap[i[1].replace(' ', '')] = i[0].replace(' ', '')  # Replace the name of each province from Chinese to English
    data.append(i[2021-Year])

# Change units to 10^7
data_new = [x/1000 for x in data]
List = [list(z) for z in zip(province, data_new)]

# Generate figure
c = (
    Map(init_opts=opts.InitOpts(width="1000px", height="600px"))  # Pixel size
        .set_global_opts(
        title_opts=opts.TitleOpts(title="China Car"),   # Title
        visualmap_opts=opts.VisualMapOpts(
            min_=0,     # Minimum value of bar
            max_=3,     # Maximum value of bar
            is_piecewise=False,  # Define the legend as segmented
            pos_top="middle",  # Position of bar
            pos_left="left",
            orient="vertical",
            range_color=['#5FACFC', '#22C2DA', '#63D5B3', '#D4EC5A', '#FFB64D', '#FA816E', '#D15B7E', '#D15B7E', '#D15B7E']  # Set the color of bar
        ),
        tooltip_opts=opts.TooltipOpts(is_show=False),   # Turn off the prompt box
        legend_opts=opts.LegendOpts(is_show=False),     # Turn off Legend
    )
        .add("Car", List, maptype="china", name_map=namemap)

        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))  # Turn off name display
        .render(Name)   # Filename
)
