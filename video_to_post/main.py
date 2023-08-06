import argparse
import configparser
import logging

import openai
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage
from moviepy.editor import VideoFileClip

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class AudioExtractor:
    def extract(self, video_file_path):
        video_clip = VideoFileClip(video_file_path)
        audio_file_path = video_file_path.replace(".mp4", ".wav")
        video_clip.audio.write_audiofile(audio_file_path)
        return audio_file_path


class WhisperTranscriber:
    def __init__(self, language):
        # language: ISO-639-1 format
        self.language = language

    def transcribe(self, audio_file_name):
        with open(audio_file_name, "rb") as f:
            transcription = openai.Audio.transcribe(
                "whisper-1", f, language=self.language
            )
            command_text = transcription.get("text")
            logging.info("Transcribed command text: %s", command_text)
            return command_text


logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class BlogPostGenerator:
    def __init__(self, config_path):
        config = configparser.ConfigParser()
        config.read_dict(
            {
                "settings": {
                    "chat_model": "gpt-3.5-turbo",
                    "system_prompt": "You are a helpful assistant.",
                    "temperature": "0.7",
                    "tools": "serpapi",
                },
            }
        )
        config.read(config_path)
        openai.api_key = config.get("api", "openai_api_key")
        system_prompt = SystemMessage(content=config.get("settings", "system_prompt"))
        agent_kwargs = {
            "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
            "system_message": system_prompt,
        }
        memory = ConversationBufferMemory(memory_key="memory", return_messages=True)
        self.chat = ChatOpenAI(
            model=config.get("settings", "chat_model"),
            temperature=float(config.get("settings", "temperature")),
            openai_api_key=config.get("api", "openai_api_key"),
        )
        self.tools = load_tools(config.get("settings", "tools").split(","))
        self.agent = initialize_agent(
            self.tools,
            self.chat,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            agent_kwargs=agent_kwargs,
            memory=memory,
        )

    def generate(self, transcription):
        """
        Generates a blog post based on the given transcription.

        :param transcription: transcribed text from the video
        :return: blog post in markdown format
        """
        # Use the transcription as the input to the agent
        response = self.agent.run(transcription)
        # The response should be in markdown format
        return response["content"]


def main():
    parser = argparse.ArgumentParser(description="Convert a video into a blog post.")
    parser.add_argument("input", help="Path to the input video file")
    parser.add_argument("output", help="Path to the output md file")
    args = parser.parse_args()

    # Extract audio from the video
    audio_extractor = AudioExtractor()
    audio_file = audio_extractor.extract(args.input)

    # Transcribe the audio to text
    whisper_transcriber = WhisperTranscriber("ko")
    transcribed_text = whisper_transcriber.transcribe(audio_file)

    # Generate a blog post from the transcribed text
    blog_post_generator = BlogPostGenerator()
    blog_post = blog_post_generator.generate(transcribed_text)

    # Write the blog post to the output file
    with open(args.output, "w") as output_file:
        output_file.write(blog_post)


if __name__ == "__main__":
    main()
