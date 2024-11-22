import os
import openai

if __name__ == "__main__":

    openai_api_key = os.environ['OPENAI_KEY']
    openai_base_url = os.environ['OPENAI_BASE_URL']

    client = openai.OpenAI(api_key=openai_api_key, base_url=openai_base_url)

    chat_completion = client.chat.completions.create(messages=[{
                                                    "role": "user",
                                                    "content": "What's the most import element for life?",
                    }],
        model = "gpt-4o"
    )

    print(chat_completion.choices[0].message.content)