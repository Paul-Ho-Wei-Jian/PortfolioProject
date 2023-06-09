import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import streamlit as st
import matplotlib.lines as lines

df = pd.read_csv("amazon_prime_titles.csv")

st.set_page_config(page_title="PortfolioProject", layout="wide")

header_spacer1, header_1, header_spacer2 = st.columns((2, 6, 2))

with header_1:
    st.header("Python-based Data Visualisation Portfolio Project")
    header_1.markdown("In this project, I showcase my data wrangling skills by applying exploratory data analysis (EDA) to an Amazon Prime dataset that I found on [Kaggle](https://www.kaggle.com/datasets/shivamb/amazon-prime-movies-and-tv-shows). The dataset contains information about amazon prime's selection of videos with it's ratings, release year, country of production, etc")
    header_1.markdown("Next, I performed EDA to gain insights into the dataset. I used python libraries such as matplotlib, seaborn to create easy to interpret visualisations. I have included a step by step visualisations, consisting of a brief insight of what I have discovered. I hope my visualisation helps you have a better understanding of Amazon prime's content structure especially if you a subscriber. :smiley:")
    

df['Count'] = 1
df['First_Country'] = df['country'].astype(str).str.split(",")
df['First_Country'] = df['First_Country'].str[0]
df['First_Country'].value_counts()

# Reduce countries's name length
df['First_Country'].replace('United States', 'USA', inplace=True)
df['First_Country'].replace('United Kingdom', 'UK',inplace=True)
df['First_Country'].replace('South Korea', 'S. Korea',inplace=True)

data = df[~(df['First_Country'] == 'nan')]
data = data.groupby('First_Country')['Count'].sum().sort_values(ascending = False)[:10]

movie_type = df.groupby('type')['type'].count()
y = len(df)
percent = ((movie_type/y)).round(2)

mf_ratio = percent.to_frame().T

left, row0_1, right = st.columns((2, 3, 2))

with row0_1:
    st.subheader("1: Movie & TV Show distribution")
    fig, ax = plt.subplots(1,1)

    ax.barh(mf_ratio.index, mf_ratio['Movie'], 
            color='#00A8E1', alpha=0.9, label='Male')
    ax.barh(mf_ratio.index, mf_ratio['TV Show'], left=mf_ratio['Movie'], 
            color='#BFF5FD', alpha=0.9, label='Female')

    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    for i in mf_ratio.index:
        ax.annotate(f"{int(mf_ratio['Movie'][i]*100)}%", 
                       xy=(mf_ratio['Movie'][i]/2, i),
                       va = 'center', ha='center',fontsize=40, fontweight='light', fontfamily='serif',
                       color='white')

        ax.annotate("Movie", 
                       xy=(mf_ratio['Movie'][i]/2, -0.25),
                       va = 'center', ha='center',fontsize=15, fontweight='light', fontfamily='serif',
                       color='white')

    for i in mf_ratio.index:
        ax.annotate(f"{int(mf_ratio['TV Show'][i]*100)}%", 
                       xy=(mf_ratio['Movie'][i]+mf_ratio['TV Show'][i]/2, i),
                       va = 'center', ha='center',fontsize=30, fontweight='light', fontfamily='serif',
                       color='white')
        ax.annotate("TV Show", 
                       xy=(mf_ratio['Movie'][i]+mf_ratio['TV Show'][i]/2, -0.25),
                       va = 'center', ha='center',fontsize=15, fontweight='light', fontfamily='serif',
                       color='white')

    for s in ['top', 'left', 'right', 'bottom']:
        ax.spines[s].set_visible(False)

    ax.legend().set_visible(False)
    st.pyplot(fig)
    
    st.markdown("We see vastly more movies than TV shows on the platform. We would do a deeper analysis on how content is being split in each country below") 
    
st.markdown("***")   
row1_space1, row1_1, row1_space2, row1_2, row1_space3 = st.columns(
    (0.1, 0.9, 0.1, 0.9, 0.1))

with row1_1:
    st.subheader("2: Top 10 countries on Amazon Prime")
    color_map = ['#f5f5f1' for _ in range(10)]
    color_map[0] = color_map[1] = color_map[2] =  '#00A8E1' 
    # color highlight

    fig, ax = plt.subplots(1,1)
    ax.bar(data.index, data, width=0.5, 
           edgecolor='darkgray',
           linewidth=0.6,color=color_map)

    #annotations
    for i in data.index:
        ax.annotate(f"{data[i]}", 
                       xy=(i, data[i] + 150), 
                       va = 'center', ha='center',fontweight='light', fontfamily='serif')

    # Remove border from plot
    for s in ['top','bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)

    # Tick labels
    ax.set_xticklabels(data.index, fontfamily='serif', rotation=0)

    ax.grid(axis='y', linestyle='-', alpha=0.4)   

    grid_y_ticks = np.arange(0, 4000, 500) # y ticks, min, max, then step
    ax.set_yticks(grid_y_ticks)
    ax.set_axisbelow(True)

    # thicken the bottom line if you want to
    plt.axhline(y = 0, color = 'black', linewidth = 1.3, alpha = .7)

    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.tick_params(axis=u'both', which=u'both',length=0)
    st.pyplot(fig)
    st.markdown("The three most frequent countries have been highlighted. The biggest content producers are primarily USA, India and UK with the rest of the country with a significant distance (Could be due to incomplete dataset or biased data collection methods)")

with row1_2:
    df = df[~(df['First_Country'] == 'nan')]
    st.subheader("3: Top 10 countries Movie & TV Show split")
    country_order = df['First_Country'].value_counts()[:11].index
    data_q2q3 = df[['type', 'First_Country']].groupby('First_Country')['type'] \
                .value_counts().unstack().loc[country_order]
    data_q2q3['sum'] = data_q2q3.sum(axis=1)
    data_q2q3_ratio = (data_q2q3.T / data_q2q3['sum']).T[['Movie', 'TV Show']]\
                      .sort_values(by='Movie',ascending=False)[::-1]
    
    fig, ax = plt.subplots(1,1)
    
    ax.barh(data_q2q3_ratio.index, data_q2q3_ratio['Movie'], 
            color='#00A8E1', alpha=0.8, label='Movie')
    ax.barh(data_q2q3_ratio.index, data_q2q3_ratio['TV Show'], left=data_q2q3_ratio['Movie'], 
            color='#221f1f', alpha=0.8, label='TV Show')

    ax.set_xticks([])
    ax.set_yticklabels(data_q2q3_ratio.index, fontfamily='serif', fontsize=11)

    # Annotation - Percentage number on the barchart
    for i in data_q2q3_ratio.index:
        ax.annotate(f"{data_q2q3_ratio['Movie'][i]*100:.3}%", 
                   xy=(data_q2q3_ratio['Movie'][i]/2, i),
                   va = 'center', ha='center',fontsize=12, fontweight='light', fontfamily='serif',
                   color='white')

    for i in data_q2q3_ratio.index:
        ax.annotate(f"{data_q2q3_ratio['TV Show'][i]*100:.3}%", 
                   xy=(data_q2q3_ratio['Movie'][i]+data_q2q3_ratio['TV Show'][i]/2, i),
                   va = 'center', ha='center',fontsize=12, fontweight='light', fontfamily='serif',
                   color='white')

    #fig.text(0.13, 0.93, 'Top 10 countries Movie & TV Show split', fontsize=15, fontweight='bold',              fontfamily='serif')
    for s in ['top', 'left', 'right', 'bottom']:
        ax.spines[s].set_visible(False)
    
    # Legend on the top right hand corner
    fig.text(0.7,0.9,"Movie", fontweight="bold", fontfamily='serif', fontsize=15, color='#00A8E1')
    fig.text(0.81,0.9,"|", fontweight="bold", fontfamily='serif', fontsize=15, color='black')
    fig.text(0.82,0.9,"TV Show", fontweight="bold", fontfamily='serif', fontsize=15,                   color='#221f1f')

    st.pyplot(fig)
    st.markdown("Most of the country's content is made up of mostly Movies, perhaps the industry main focus is on movies and not TV Shows. All 6 pieces of content from Australia is movies ")
    st.markdown("On the other hand Spain and Japan has more TV shows than movies. I am a big fan of anime content and it is pretty popular in Japan, it makes sense that they have more TV shows than movies")
    
st.markdown("***")
row2_space1, row2_1, row2_space2, row2_2, row2_space3 = st.columns((0.1, 1, 0.1, 1, 0.1))
# Taken from: https://www.primevideo.com/help/ref=atv_hp_nd_cnt?nodeId=GFGQU3WYEG6FSJFJ
ratings_ages = {
    'TV-PG': 'Older Kids',
    'TV-MA': 'Adults',
    'TV-Y7-FV': 'Older Kids',
    'TV-Y7': 'Older Kids',
    'TV-14': 'Teens',
    'R': 'Adults',
    'TV-Y': 'Kids',
    'NR': 'Adults',
    'PG-13': 'Teens',
    'TV-G': 'Kids',
    'PG': 'Older Kids',
    'G': 'Kids',
    '13+': 'Teens',
    '16+': 'Young Adults',
    '18+' : 'Adults',
    '7+' : 'Older Kids',
    'ALL' : 'Kids',
    'NOT_RATE': 'Adults'
}

df['target_ages'] = df['rating'].replace(ratings_ages)
data = df.groupby('First_Country')[['First_Country', 'Count']].sum().sort_values(by = 'Count', ascending = False).reset_index()[:10]
data = data['First_Country']

df_heatmap = df.loc[df['First_Country'].isin(data)]
df_heatmap = pd.crosstab(df_heatmap['First_Country'],df_heatmap['target_ages'],normalize = "index").T

with row2_1:
    st.subheader("4: Does Amazon prime target a certain demographics, is there a variation based on the country")
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    country_order2 = ['USA', 'India', 'UK', 'Canada', 
 'Italy', 'Spain', 'Germany', 
 'France', 'Australia']

    age_order = ['Kids', 'Older Kids', 'Teens', 'Young Adults', 'Adults']

    sns.heatmap(df_heatmap.loc[age_order,country_order2],square=True, linewidth=2.5,cbar=False,
                annot=True,fmt='1.0%',vmax=.6,vmin=0.05,ax=ax,annot_kws={"fontsize":12}, cmap = 'Blues')

    ax.spines['top'].set_visible(False)

    # fig.text(.4, .45, 'Target ages proportion of total content by country', 
             # fontweight='bold', fontfamily='serif', fontsize=15,ha='right')   

    ax.set_yticklabels(ax.get_yticklabels(), fontfamily='serif', rotation = 0, fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), fontfamily='serif', rotation=90, fontsize=11)

    ax.set_ylabel('')    
    ax.set_xlabel('')
    ax.tick_params(axis=u'both', which=u'both',length=0)
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown(" The heatmap above shows which segment of audience does the content in each country targets. Older kids and kids does not seem to be the target audience for all the countries. Interestingly, India, Italy, France has a high percentage of content created for teens")
    
with row2_2:
    st.subheader("5: Movies & TV Shows added over time")
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    color = ['#00A8E1', '#BFF5FD']

    for i, mtv in enumerate(df['type'].value_counts().index):
        mtv_rel = df[df['type']==mtv]['release_year'].value_counts().sort_index()
        ax.plot(mtv_rel.index, mtv_rel, color=color[i], label=mtv)
        ax.fill_between(mtv_rel.index, 0, mtv_rel, color=color[i], alpha=0.9)

    ax.yaxis.tick_right()
    ax.axhline(y = 0, color = 'black', linewidth = 1.3, alpha = .7)

    # Setting x axis limit 
    ax.set_xlim(2008,2020)
    plt.xticks(np.arange(2000, 2021, 1))

    for s in ['top', 'right','bottom','left']:
        ax.spines[s].set_visible(False)

    #fig.text(0.13, 0.85, 'Movies & TV Shows added over time', fontsize=15, fontweight='bold', fontfamily='serif')
    fig.text(0.13,0.28,"Movie", fontweight="bold", fontfamily='serif', fontsize=15,                              color='#00A8E1')
    fig.text(0.19,0.28,"|", fontweight="bold", fontfamily='serif', fontsize=15, color='black')
    fig.text(0.2,0.28,"TV Show", fontweight="bold", fontfamily='serif', fontsize=15,                            color='#BFF5FD')

    ax.tick_params(axis=u'both', which=u'both',length=0)
    st.pyplot(fig)
    st.markdown("Tv show content has been added on a steady uptrend, whereas movie content has multiple flunctuation but generally it is still on a uptrend. There is sharp increase of movie content from 2015 ~ 2016 and 2018 ~ 2019, with content peaking around 2019")

st.markdown("***")
left, midText = st.columns((0.1, 2.5))
with midText:
    st.markdown(
        "Thanks you for going through this mini-analysis with me! You can find the python script and notebook on my [Github](https://github.com/Paul-Ho-Wei-Jian/PortfolioProject.git) :purple_heart:")
