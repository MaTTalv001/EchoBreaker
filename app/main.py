import streamlit as st
import pandas as pd
import requests
import feedparser
from io import BytesIO
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import json

# OpenAI API キーの設定
client = OpenAI(api_key=st.secrets["openai"]["api_key"])

def main():
    st.set_page_config(layout="wide", page_title="興味のないニュース")

    # サイドバーにアプリの説明を追加
    with st.sidebar:
        st.title("アプリについて")
        st.write("""
        このアプリは、あなたの興味から最も遠いニュースを見つけ出します。
        新しい視点や意外な情報との出会いを提供することを目的としています。
        """)
        st.subheader("使い方")
        st.write("""
        1. 「架空のニュースを生成」ボタンをクリックします。
        2. 生成された架空のニュースから、最も興味があるものを選択します。
        3. 「ニュース取得」ボタンをクリックして、実際のニュースを取得します。
        4. 表示されたニュースは、あなたの興味との関連性が低い順に並んでいます。
        """)
        st.subheader("関連性スコアについて")
        st.write("""
        関連性スコアは-1から1の間の値を取ります：
        - 1に近いほど：あなたの興味と非常に関連がある
        - 0に近いほど：あなたの興味とほとんど関連がない
        - 負の値：あなたの興味と逆の関連性がある可能性がある
        """)

    st.title("興味のないニュース")
    st.divider()

    # col1, col2 = st.columns(2)

    
    if st.button("架空のニュースを生成", key="generate_fake_news"):
        with st.spinner("架空のニュースを生成中..."):
            fake_news = generate_fake_news()
            st.session_state['fake_news'] = fake_news

    if 'fake_news' in st.session_state:
        
        st.subheader("生成された架空のニュース")
        selected_news = st.radio(
            "最も関心が高いニュースを選択してください：", 
            options=[news['summary'] for news in st.session_state['fake_news']],
            key="selected_news"
        )
        
        if st.button("ニュース取得", key="fetch_news"):
            context = selected_news
            news_feed_mode(context)

def generate_fake_news():
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
           {"role": "system", "content": "あなたは優秀なニュース記者です。これまでさまざまな記事を作成してきました。回答は必ずJSON形式で行ってください。"},
            {"role": "user", "content": "架空のニュースの概要を5つ生成してください。各概要は1-2文程度のわかりやすいものにしてください。出力は'summary'キーを持つJSONオブジェクトとし、その値は概要オブジェクトの配列としてください。各概要オブジェクトは'summary'キーを持ちます。"}
        ],
        response_format={"type": "json_object"},
        temperature=0.8
    )

    fake_news = json.loads(response.choices[0].message.content)
    return fake_news['summary']

def news_feed_mode(context):
    st.header("ニュース取得結果")
    st.info(f"選択された関心事項: {context}", icon="ℹ️")

    with st.spinner('関連の低いニュースを取得中...'):
        news_articles = fetch_news_articles()
        df = calculate_similarity_and_sort(context, news_articles, reverse=True)

        st.subheader("関連性の低いニュース（上位10件）")
        display_news_details(df, 10)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="ニュース一覧をCSVでダウンロード",
            data=csv,
            file_name="news_articles.csv",
            mime="text/csv",
        )

def fetch_news_articles():
    # RSSフィードのURLリスト
    rss_urls = [
        'https://business.nikkei.com/rss/sns/nb.rdf',
        # 'https://business.nikkei.com/rss/sns/nb-x.rdf',
        # 'https://business.nikkei.com/rss/sns/nb-plus.rdf',
        # 'https://xtech.nikkei.com/rss/xtech-it.rdf',
        # 'https://xtech.nikkei.com/rss/xtech-mono.rdf',
        # 'https://xtech.nikkei.com/rss/xtech-hlth.rdf',
        'https://xtech.nikkei.com/rss/index.rdf',
        'https://rss.itmedia.co.jp/rss/2.0/news_bursts.xml',
        'https://www3.nhk.or.jp/rss/news/cat0.xml',
        'https://news.yahoo.co.jp/rss/topics/top-picks.xml',
        'https://www.roomie.jp/feed/'

    ]

    news_articles = []

    def clean_html(html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup.get_text().strip()

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    for url in rss_urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            feed = feedparser.parse(BytesIO(response.content))
            
            if feed.entries:
                for entry in feed.entries:
                    article = {
                        'title': entry.get('title', ''),
                        'link': entry.get('link', ''),
                        'published': entry.get('published', '') or entry.get('updated', '')
                    }
                    
                    if 'summary' in entry:
                        article['summary'] = clean_html(entry.summary)
                    elif 'description' in entry:
                        article['summary'] = clean_html(entry.description)
                    elif 'content' in entry:
                        article['summary'] = clean_html(entry.content[0].value)
                    else:
                        article['summary'] = ''
                    
                    news_articles.append(article)
        
        except Exception as e:
            st.error(f"Error processing {url}: {str(e)}")

    return news_articles

def calculate_similarity_and_sort(context, news_articles, reverse=True):
    model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
    context_embedding = model.encode([context])[0]

    for article in news_articles:
        article_text = article['title'] + " " + article['summary']
        article_embedding = model.encode([article_text])[0]
        similarity = cosine_similarity([context_embedding], [article_embedding])[0][0]
        article['similarity_score'] = similarity

    df = pd.DataFrame(news_articles)
    return df.sort_values('similarity_score', ascending=True)  # 昇順にソート

def display_news_details(df, num_articles):
    for i, article in enumerate(df[0:num_articles].to_dict('records')):
        title = f"ニュース {i+1}: {article['title']}"
        
        # 関連性スコアに基づいて色を変更
        if article['similarity_score'] < 0:
            score_color = "red"
        elif article['similarity_score'] < 0.3:
            score_color = "green"
        else:
            score_color = "orange"
        
        with st.expander(title):
            st.write(f"タイトル: {article['title']}")
            st.write(f"公開日: {article['published']}")
            st.write(f"リンク: {article['link']}")
            st.write("要約:")
            st.write(article['summary'])
            st.markdown(f"関連性スコア: <font color='{score_color}'>{article['similarity_score']:.4f}</font>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()