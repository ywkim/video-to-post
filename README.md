# Video-to-Post

Video-to-Post는 비디오를 블로그 포스트로 변환하는 Python 패키지입니다. 이 패키지는 비디오에서 오디오를 추출하고, 음성을 텍스트로 변환하며, 필요한 경우 이미지를 추출합니다. 결과는 Markdown 형식의 파일로 출력됩니다.

## 설치

이 패키지는 pip를 사용하여 설치할 수 있습니다.

```bash
pip install video-to-post
```

또는, 이 패키지는 poetry를 사용하여 설치할 수 있습니다.

```bash
poetry add video-to-post
```

## 사용법

이 패키지는 커맨드라인에서 다음과 같이 사용할 수 있습니다.

```bash
video-to-post /path/to/video.mp4 -o /path/to/output.md
```

이 명령은 `video.mp4` 파일을 처리하고, 결과를 `output.md` 파일로 저장합니다.

## 라이선스

이 프로젝트는 MIT 라이선스 하에 라이선스가 부여됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.
