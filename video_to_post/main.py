import argparse
import configparser
import logging
import sys

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

DEFAULT_CONFIG = {
    "settings": {
        "chat_model": "gpt-3.5-turbo",
        "system_prompt": "You are a helpful assistant.",
        "temperature": "0.7",
    }
}


class AudioExtractor:
    def extract(self, video_file_path):
        video_clip = VideoFileClip(video_file_path)
        audio_file_path = video_file_path.replace(".mp4", ".m4a")
        video_clip.audio.write_audiofile(audio_file_path, codec="aac")
        return audio_file_path


class WhisperTranscriber:
    def __init__(self, language, openai_api_key):
        # language: ISO-639-1 format
        self.language = language
        openai.api_key = openai_api_key

    def transcribe(self, audio_file_name):
        with open(audio_file_name, "rb") as f:
            transcription = openai.Audio.transcribe(
                "whisper-1", f, language=self.language
            )
            command_text = transcription.get("text")
            logging.info("Transcribed command text: %s", command_text)
            return command_text


class BlogPostGenerator:
    def __init__(self, config):
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
        self.agent = initialize_agent(
            [],
            self.chat,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
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
        response_message = self.agent.run(transcription)
        # The response should be in markdown format
        return response_message


def main():
    parser = argparse.ArgumentParser(description="Convert a video into a blog post.")
    parser.add_argument("input", help="Path to the input video file")
    parser.add_argument("-o", "--output", help="Path to the output md file")
    parser.add_argument(
        "--config_file", default="config.ini", help="Path to a config file"
    )
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read_dict(DEFAULT_CONFIG)
    config.read(args.config_file)

    # Extract audio from the video
    audio_extractor = AudioExtractor()
    audio_file = audio_extractor.extract(args.input)

    # Transcribe the audio to text
    whisper_transcriber = WhisperTranscriber(
        "ko", openai_api_key=config.get("api", "openai_api_key")
    )
    transcribed_text = whisper_transcriber.transcribe(audio_file)

    # Generate a blog post from the transcribed text
    blog_post_generator = BlogPostGenerator(config)
    blog_post = blog_post_generator.generate(transcribed_text)

    # Write the blog post to the output file
    if args.output:
        with open(args.output, "w") as output_file:
            output_file.write(blog_post)
    else:
        sys.stdout.write(blog_post)


if __name__ == "__main__":
    main()
