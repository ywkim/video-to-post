"""
This module provides functionality to convert videos into blog posts.
"""

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
        "chat_model": "gpt-3.5-turbo-16k",
        "system_prompt": "You are a helpful assistant.",
        "temperature": "0.7",
    }
}

# Constants for message prompts
GENERATE_TITLE_PROMPT = (
    "다음은 발표 영상 스크립트이고, 영상에 기반해서 블로그 포스트를 쓰려고 하니, 포스트 제목과 요약문을 2~3개 문단으로 작성해:\n\n{}"
)
GENERATE_TOC_PROMPT = "이를 바탕으로, 블로그 포스트의 목차를 작성해\n- 중요도와 정량적 비중도 명시해"
GENERATE_BLOG_POST_PROMPT = "이제 전체 블로그 포스트를 Markdown 형식으로 작성해"


class AudioExtractor:
    """
    This class handles the extraction of audio from video.
    """

    def __init__(self, video_file_path):
        """
        Initialize an AudioExtractor instance.

        :param video_file_path: str
        """
        self.video_file_path = video_file_path

    def extract(self):
        """
        Extracts audio from the video file and saves it to an audio file.

        :return: str
        """
        video_clip = VideoFileClip(self.video_file_path)
        audio_file_path = self.video_file_path.replace(".mp4", ".m4a")
        video_clip.audio.write_audiofile(audio_file_path, codec="aac")
        return audio_file_path


class WhisperTranscriber:
    """
    This class handles the transcription of audio to text.
    """

    def __init__(self, language, audio_file_name, openai_api_key):
        """
        Initialize a WhisperTranscriber instance.

        :param language: str (ISO-639-1 format)
        :param audio_file_name: str
        :param openai_api_key: str
        """
        self.language = language
        self.audio_file_name = audio_file_name
        openai.api_key = openai_api_key

    def transcribe(self):
        """
        Transcribes the audio file to text.

        :return: str
        """
        with open(self.audio_file_name, "rb") as audio_file:
            transcription = openai.Audio.transcribe(
                "whisper-1", audio_file, language=self.language
            )
            command_text = transcription.get("text")
            logging.info("Transcribed command text: %s", command_text)
            return command_text


class BlogPostGenerator:
    def __init__(self, chat_model, system_prompt, temperature, openai_api_key):
        """
        Initialize a BlogPostGenerator instance.

        :param chat_model: model for chat
        :param system_prompt: initial prompt for the system
        :param temperature: controlling randomness of outputs
        :param openai_api_key: API key of OpenAI
        """
        openai.api_key = openai_api_key
        system_prompt_message = SystemMessage(content=system_prompt)
        agent_kwargs = {
            "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
            "system_message": system_prompt_message,
        }
        memory = ConversationBufferMemory(memory_key="memory", return_messages=True)
        self.chat = ChatOpenAI(
            model=chat_model,
            temperature=temperature,
            openai_api_key=openai_api_key,
        )
        self.agent = initialize_agent(
            [],
            self.chat,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            agent_kwargs=agent_kwargs,
            memory=memory,
        )

    def generate_title_and_summary(self, transcription):
        """
        Generates a blog post title and summary based on the given transcription.

        :param transcription: transcribed text from the video
        :return: title and summary of the blog post
        """
        response_message = self.agent.run(GENERATE_TITLE_PROMPT.format(transcription))
        return response_message

    def generate_toc(self):
        """
        Generates a table of contents for the blog post.

        :return: table of contents for the blog post
        """
        response_message = self.agent.run(GENERATE_TOC_PROMPT)
        return response_message

    def generate_blog_post(self):
        """
        Generates the full blog post.

        :return: full blog post in markdown format
        """
        response_message = self.agent.run(GENERATE_BLOG_POST_PROMPT)
        # The response should be in markdown format
        return response_message

    def generate(self, transcription):
        """
        Generates a blog post based on the given transcription.

        :param transcription: transcribed text from the video
        :return: blog post in markdown format
        """
        # Generate a title and summary for the blog post
        title_and_summary = self.generate_title_and_summary(transcription)

        # Generate a table of contents for the blog post
        toc = self.generate_toc()

        # Generate the full blog post
        blog_post = self.generate_blog_post()

        return blog_post


def main():
    """
    The main execution point of the script from the command line.
    """
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
    audio_extractor = AudioExtractor(args.input)
    audio_file_path = audio_extractor.extract()

    # Transcribe the audio to text
    whisper_transcriber = WhisperTranscriber(
        "ko", audio_file_path, config.get("api", "openai_api_key")
    )
    transcribed_text = whisper_transcriber.transcribe()

    # Generate a blog post from the transcribed text
    blog_post_generator = BlogPostGenerator(
        chat_model=config.get("settings", "chat_model"),
        system_prompt=config.get("settings", "system_prompt"),
        temperature=float(config.get("settings", "temperature")),
        openai_api_key=config.get("api", "openai_api_key"),
    )
    blog_post = blog_post_generator.generate(transcribed_text)

    # Write the blog post to the output file
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(blog_post)
    else:
        sys.stdout.write(blog_post)


if __name__ == "__main__":
    main()
