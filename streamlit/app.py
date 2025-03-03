import streamlit as st
import requests
import matplotlib.pyplot as plt

# Define categories (for reference)
CATEGORIES = [
    'Arts', 'Automobile', 'Bank', 'Blog', 'Business', 'Crime', 'Economy', 'Education',
    'Entertainment', 'Health', 'Politics', 'Society', 'Sports', 'Technology', 'Tourism', 'World'
]

# Function to create probability chart
def create_probability_chart(probabilities):
    sorted_probs = dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True))
    categories = list(sorted_probs.keys())[:5]  # Top 5
    probs = list(sorted_probs.values())[:5]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(categories, probs, color='skyblue')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.2%}', ha='center', va='bottom')
    
    plt.title('Top Category Probabilities')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    return fig

def main():
    st.title("News Category Classifier")
    st.write("Enter a news article to classify it into one of the 16 categories.")
    
    # Text input
    news_text = st.text_area("Enter news text:", height=200)
    example_button = st.button("Load Example Text")
    
    # Example text
    example_text = """
    The European Central Bank cut interest rates on Thursday for the third time since June, 
    as the euro zone's economy continues to struggle and inflation edges closer to target. 
    The ECB lowered its benchmark deposit rate by 25 basis points to 3.25%, 
    in line with market expectations.
    """
    
    if example_button:
        news_text = example_text
        st.session_state.news_text = news_text
        st.experimental_rerun()
    
    # Classify button
    classify_button = st.button("Classify News")
    
    if news_text and classify_button:
        if len(news_text.strip()) < 10:
            st.warning("Please enter a longer news text for better classification.")
        else:
            with st.spinner("Classifying..."):
                # Send request to FastAPI
                response = requests.post("comfortable-jodi-nikhil364-e49440d6.koyeb.app/", json={"text": news_text})
                
                if response.status_code == 200:
                    result = response.json()
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        predicted_category = result["predicted_category"]
                        probabilities = result["probabilities"]
                        
                        # Display results
                        st.success(f"Predicted Category: **{predicted_category}**")
                        
                        # Top 3 categories
                        st.write("### Top 3 Categories:")
                        top3 = dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3])
                        for category, prob in top3.items():
                            st.write(f"- {category}: {prob:.2%}")
                        
                        # Probability chart
                        st.write("### Probability Distribution:")
                        chart = create_probability_chart(probabilities)
                        st.pyplot(chart)
                        
                        # Confidence info
                        top_prob = max(probabilities.values())
                        if top_prob > 0.80:
                            st.info("High confidence prediction ✅")
                        elif top_prob > 0.50:
                            st.info("Moderate confidence prediction ⚠️")
                        else:
                            st.warning("Low confidence prediction ⚠️")
                else:
                    st.error("Error classifying text")
    
    # Info section
    with st.expander("About this app"):
        st.write("""
        This app uses a FastAPI backend with a Hugging Face model to classify news articles into 16 categories.
        See the full list of categories above.
        """)

if __name__ == "__main__":
    main()
