import streamlit as st
import requests



@st.cache_resource
def load_recommender():
    from src.recommender import (
        recommend,
        recommend_from_index,
        recommend_from_query,
        df
    )

    return (
        recommend,
        recommend_from_index,
        recommend_from_query,
        df
    )


recommend, recommend_from_index, recommend_from_query, df = load_recommender()

st.set_page_config(
    page_title="WeebWise",
    page_icon="🌸",
    layout="wide"
)

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(255, 77, 109, 0.18),
                transparent 25%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(124, 92, 255, 0.20),
                transparent 25%
            ),
            radial-gradient(
                circle at 50% 90%,
                rgba(0, 210, 255, 0.12),
                transparent 30%
            ),
            #090b13;
        color: #f5f5f7;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 3rem;
        padding-bottom: 4rem;
    }

    .hero-title {
        text-align: center;
        font-size: 52px;
        font-weight: 800;
        margin-bottom: 10px;
        background: linear-gradient(
            90deg,
            #ff4d6d,
            #ff8a5b,
            #ffd166,
            #8b7cff,
            #38d9ff
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        text-align: center;
        color: #b9bdc9;
        font-size: 18px;
        margin-bottom: 35px;
    }

    .section-title {
        font-size: 30px;
        font-weight: 750;
        margin-top: 35px;
        margin-bottom: 20px;
    }

    .rank {
        color: #ffd166;
        font-size: 14px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .anime-title {
        font-size: 19px;
        font-weight: 750;
        color: white;
        margin-top: 12px;
        margin-bottom: 8px;
        line-height: 1.3;
        min-height: 50px;
    }

    .score {
        display: inline-block;
        background: linear-gradient(
            90deg,
            #ff4d6d,
            #8b5cf6
        );
        color: white;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .anime-info {
        margin-top: 12px;
        padding: 10px;
        border-radius: 10px;
        background: rgba(255,255,255,0.04);
        color: #d8dbe5;
        font-size: 13px;
        line-height: 1.8;
        min-height: 105px;
    }

    .metadata {
        color: #b9bdc9;
        font-size: 12px;
        line-height: 1.55;
        margin-top: 8px;
        min-height: 20px;
    }

    .reason {
        color: #c9ccd6;
        font-size: 13px;
        line-height: 1.5;
        margin-top: 14px;
        min-height: 40px;
    }

    .poster {
        width: 100%;
        height: 360px;
        object-fit: cover;
        border-radius: 12px;
    }

    .poster-placeholder {
        width: 100%;
        height: 360px;
        display: flex;
        align-items: center;
        justify-content: center;
        background:
            linear-gradient(
                135deg,
                #ff4d6d,
                #8b5cf6,
                #38d9ff
            );
        border-radius: 12px;
        font-size: 60px;
    }

    .footer {
        text-align: center;
        color: #777c8c;
        margin-top: 50px;
        font-size: 13px;
    }

    div.stButton > button {
        background: linear-gradient(
            90deg,
            #ff4d6d,
            #8b5cf6
        );
        color: white;
        border: none;
        border-radius: 12px;
        padding: 10px 25px;
        font-weight: 700;
    }

    div.stButton > button:hover {
        box-shadow:
            0 8px 20px rgba(139, 92, 246, 0.35);
    }

    div[data-baseweb="input"] {
        border-radius: 12px;
    }

    div[data-baseweb="select"] > div {
        border-radius: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


if "status" not in st.session_state:
    st.session_state.status = None

if "data" not in st.session_state:
    st.session_state.data = None

if "user_input" not in st.session_state:
    st.session_state.user_input = ""


@st.cache_data(show_spinner=False)
def get_jikan_image(title):

    try:

        response = requests.get(
            "https://api.jikan.moe/v4/anime",
            params={
                "q": title,
                "limit": 1
            },
            timeout=10
        )

        if response.status_code == 200:

            results = response.json().get(
                "data",
                []
            )

            if results:

                images = results[0].get(
                    "images",
                    {}
                )

                jpg = images.get(
                    "jpg",
                    {}
                )

                return (
                    jpg.get("large_image_url")
                    or jpg.get("image_url")
                )

    except Exception:
        pass

    return None


def get_image(idx, title):

    possible_columns = [
        "image_url",
        "image",
        "poster_url",
        "images_jpg_image_url",
        "large_image_url"
    ]

    if idx is not None:

        try:

            row = df.iloc[idx]

            for column in possible_columns:

                if column in df.columns:

                    value = row[column]

                    if (
                        value is not None
                        and str(value).strip() != ""
                        and str(value) != "nan"
                    ):
                        return str(value)

        except Exception:
            pass

    return get_jikan_image(title)


def find_anime_index(title):

    title = title.lower().strip()

    try:

        matches = df[
            (df["title"] == title)
            |
            (df["title_english"] == title)
        ]

        if not matches.empty:
            return matches.index[0]

    except Exception:
        pass

    return None


def unpack_recommendation(item):

    if len(item) == 3:
        return item

    if len(item) == 2:

        anime, score = item

        return (
            anime,
            score,
            []
        )

    return (
        item[0],
        0,
        []
    )


def display_recommendations(recommendations):

    st.markdown(
        '<div class="section-title">✨ Top Recommendations</div>',
        unsafe_allow_html=True
    )

    for start in range(
        0,
        len(recommendations),
        4
    ):

        batch = recommendations[
            start:start + 4
        ]

        cols = st.columns(
            4,
            gap="medium"
        )

        for position, item in enumerate(batch):

            anime, score, reasons = (
                unpack_recommendation(item)
            )

            with cols[position]:

                idx = find_anime_index(
                    anime
                )

                image = get_image(
                    idx,
                    anime
                )

                row = None

                if idx is not None:

                    try:
                        row = df.iloc[idx]
                    except Exception:
                        row = None

                with st.container(
                    border=True
                ):

                    st.markdown(
                        f"""
                        <div class="rank">
                            #{start + position + 1}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    if image:

                        st.image(
                            image,
                            use_container_width=True
                        )

                    else:

                        st.markdown(
                            """
                            <div class="poster-placeholder">
                                🎌
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    st.markdown(
                        f"""
                        <div class="anime-title">
                            {anime}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f"""
                        <span class="score">
                            ⭐ {score:.2f} match
                        </span>
                        """,
                        unsafe_allow_html=True
                    )

                    if row is not None:

                        anime_type = str(
                            row.get(
                                "type",
                                ""
                            )
                        )

                        episodes = str(
                            row.get(
                                "episodes",
                                ""
                            )
                        )

                        duration = str(
                            row.get(
                                "duration",
                                ""
                            )
                        )

                        status_value = str(
                            row.get(
                                "status",
                                ""
                            )
                        )

                        source = str(
                            row.get(
                                "source",
                                ""
                            )
                        )

                        studios = str(
                            row.get(
                                "studios",
                                ""
                            )
                        )

                        genres = str(
                            row.get(
                                "genres",
                                ""
                            )
                        )

                        themes = str(
                            row.get(
                                "themes",
                                ""
                            )
                        )

                        info_items = []

                        if (
                            anime_type
                            and anime_type != "nan"
                        ):
                            info_items.append(
                                f"📺 {anime_type}"
                            )

                        if (
                            episodes
                            and episodes != "nan"
                            and episodes != "0"
                        ):
                            info_items.append(
                                f"🎬 {episodes} episodes"
                            )

                        if (
                            duration
                            and duration != "nan"
                        ):
                            info_items.append(
                                f"⏱️ {duration}"
                            )

                        if (
                            status_value
                            and status_value != "nan"
                        ):
                            info_items.append(
                                f"📅 {status_value}"
                            )

                        if info_items:

                            st.markdown(
                                f"""
                                <div class="anime-info">
                                    {"<br>".join(info_items)}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        if (
                            genres
                            and genres != "nan"
                        ):

                            st.markdown(
                                f"""
                                <div class="metadata">
                                    <b style="
                                        color:#ff8a5b;
                                    ">
                                        🎭 Genres:
                                    </b>
                                    {genres}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        if (
                            themes
                            and themes != "nan"
                        ):

                            st.markdown(
                                f"""
                                <div class="metadata">
                                    <b style="
                                        color:#8b7cff;
                                    ">
                                        🎨 Themes:
                                    </b>
                                    {themes}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        if (
                            studios
                            and studios != "nan"
                        ):

                            st.markdown(
                                f"""
                                <div class="metadata">
                                    <b style="
                                        color:#38d9ff;
                                    ">
                                        🏢 Studio:
                                    </b>
                                    {studios}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        if (
                            source
                            and source != "nan"
                        ):

                            st.markdown(
                                f"""
                                <div class="metadata">
                                    <b style="
                                        color:#ffd166;
                                    ">
                                        📚 Source:
                                    </b>
                                    {source}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                    if reasons:

                        reason_text = (
                            " • ".join(reasons)
                        )

                        st.markdown(
                            f"""
                            <div class="reason">
                                <b style="
                                    color:#ff8a5b;
                                ">
                                    Why recommended:
                                </b>
                                {reason_text}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )


st.markdown(
    """
    <div class="hero-title">
        🌸 WeebWise
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-subtitle">
        Discover your next favorite anime using
        semantic and content-based recommendations.
    </div>
    """,
    unsafe_allow_html=True
)


with st.form("recommend_form"):

    user_input = st.text_input(
        "🔍 What do you want to watch?",
        placeholder=(
            "Enter an anime title or describe "
            "what you're looking for..."
        ),
        value=st.session_state.user_input
    )

    submitted = st.form_submit_button(
        "Recommend Anime"
    )


if submitted:

    if not user_input.strip():

        st.warning(
            "Please enter an anime title or description."
        )

    else:

        st.session_state.user_input = (
            user_input
        )

        with st.spinner(
            "🧠 Finding the best recommendations..."
        ):

            status, data = recommend(
                user_input
            )

        st.session_state.status = status
        st.session_state.data = data

        st.rerun()


status = st.session_state.status
data = st.session_state.data


if status == "not_found":

    st.info(
        "🔎 No exact anime title found. "
        "Using semantic search based on your description..."
    )

    with st.spinner(
        "🧠 Searching semantically..."
    ):

        recommendations = (
            recommend_from_query(
                st.session_state.user_input
            )
        )

    display_recommendations(
        recommendations
    )


elif status == "fuzzy":

    st.markdown(
        '<div class="section-title">🔎 Did you mean?</div>',
        unsafe_allow_html=True
    )

    suggestions = [
        suggestion[0]
        for suggestion in data
    ]

    selected_title = st.selectbox(
        "Choose an anime",
        suggestions,
        key="fuzzy_selection"
    )

    if st.button(
        "Recommend",
        key="fuzzy_recommend"
    ):

        selected = next(
            suggestion
            for suggestion in data
            if suggestion[0] == selected_title
        )

        selected_index = find_anime_index(
            selected[0]
        )

        if selected_index is not None:

            with st.spinner(
                "🧠 Finding similar anime..."
            ):

                recommendations = (
                    recommend_from_index(
                        selected_index
                    )
                )

            st.session_state.status = (
                "success"
            )

            st.session_state.data = (
                recommendations
            )

            st.rerun()


elif status == "multiple":

    st.markdown(
        '<div class="section-title"> Multiple matches found</div>',
        unsafe_allow_html=True
    )

    options = []

    for idx in data.index:

        row = data.loc[idx]

        title = (
            row["title_english"]
            if row["title_english"] != ""
            else row["title"]
        )

        options.append(
            (title, idx)
        )

    selected_title = st.selectbox(
        "Which anime did you mean?",
        [
            title
            for title, idx in options
        ],
        key="multiple_selection"
    )

    selected_index = next(
        idx
        for title, idx in options
        if title == selected_title
    )

    if st.button(
        "Recommend",
        key="multiple_recommend"
    ):

        with st.spinner(
            "🧠 Finding similar anime..."
        ):

            recommendations = (
                recommend_from_index(
                    selected_index
                )
            )

        st.session_state.status = (
            "success"
        )

        st.session_state.data = (
            recommendations
        )

        st.rerun()


elif status == "success":

    display_recommendations(
        data
    )


st.markdown(
    """
    <div class="footer">
        Built with Python • TF-IDF • Sentence-BERT
        • Cosine Similarity • MMR • WeebWise AI
    </div>
    """,
    unsafe_allow_html=True
)