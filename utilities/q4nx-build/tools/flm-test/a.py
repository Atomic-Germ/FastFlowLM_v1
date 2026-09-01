import base64
from openai import OpenAI

client = OpenAI(base_url="http://localhost:52625/v1", api_key="none")
MODEL = "gemma4-it:e2b"
MODEL = "qwen3:0.6b"

def encode_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def test_images():
    print("=== Image Test ===")
    # ice_cream_b64 = encode_file("iceCream.jpg")
    # panda_b64 = encode_file("panda.png")
    # german_b64 = encode_file("german.png")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe both of these images."},
                    # {
                    #     "type": "image_url",
                    #     "image_url": {"url": f"data:image/jpeg;base64,{panda_b64}"},
                    # },
                    # {
                    #     "type": "image_url",
                    #     "image_url": {"url": f"data:image/png;base64,{german_b64}"},
                    # },
                ],
            }
        ],
    )
    print(response.choices[0].message.content)


def test_audio():
    print("\n=== Audio Test ===")
    audio_b64 = encode_file("nvidia.mp3")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe what you hear in this audio."},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64},
                    },
                ],
            }
        ],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    test_images()
    # test_audio()
