import pandas as pd
from pyecharts.charts import Map
from pyecharts import options as opts

df = pd.read_excel('ChinaVehicleOwnership.xlsx', sheet_name='Sheet1')#读取数据
vals = df.values#获得数据

province = []#省份
data = []#数值
namemap = {}#名称映射字典

for i in vals[0:]:
    province.append(i[0].replace(' ', ''))#去除字符串中的空格，省份名应当是映射后的英文拼音
    namemap[i[1].replace(' ', '')] = i[0].replace(' ', '')#映射关系为第一列的中文对应第零列的英文拼音
    data.append(i[4])#第2开始是2020，4是2018，8是2014，9是2013,12是2010

#单位改为10^7
data_new = [x/1000 for x in data]

list = [list(z) for z in zip(province, data_new)]
#print(list)

# 软件工程专业
c = (
    Map(init_opts=opts.InitOpts(width="1000px", height="600px"))  # 可切换主题
        .set_global_opts(
        title_opts=opts.TitleOpts(title="China Car"), #图标题
        visualmap_opts=opts.VisualMapOpts(
            min_=0, # bar的最小值
            max_=3,  #bar的最大值
            is_piecewise=False,  # 定义图例为分段型，默认为连续的图例
            pos_top="middle",  # bar位置
            pos_left="left",
            orient="vertical",
            range_color=['#5FACFC', '#22C2DA', '#63D5B3', '#D4EC5A', '#FFB64D', '#FA816E', '#D15B7E', '#D15B7E', '#D15B7E']  #设置bar的颜色
        ),
        tooltip_opts=opts.TooltipOpts(is_show=False),   # 关闭提示框
        legend_opts=opts.LegendOpts(is_show=False), # 关闭图例
    )
        .add("Car", list, maptype="china",name_map=namemap)
        #.set_global_opts(
        #title_opts=opts.TitleOpts(title=""),
        #visualmap_opts=opts.VisualMapOpts(max_=1200, ),
        #)  #显示上图
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))  #关闭名称显示
        .render("Car2018.html") #文件名
)


