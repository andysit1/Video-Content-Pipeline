
import os
import sys
import subprocess

import google.auth
import googleapiclient.discovery
import googleapiclient.errors



def youtube_upload(video_file_path, client_secret_file, scopes, api_service_name, api_version, video_title, video_description):
    # Load the client secrets file
    creds = google.oauth2.credentials.Credentials.from_authorized_user_info(info=client_secret_file, scopes=scopes)

    # Create the YouTube API client
    youtube = googleapiclient.discovery.build(api_service_name, api_version, credentials=creds)

    # Get the video file size
    video_size = os.path.getsize(video_file_path)

    # Set the metadata for the video
    body = {
        'snippet': {
            'title': video_title,
            'description': video_description,
            'tags': get_tags(video_file_path),  # Get tags from video or input file
            'categoryId': 28
        },
        'status': {
            'privacyStatus': 'private'
        }
    }

    # Create the request to upload the video
    request = youtube.videos().insert(part=','.join(body.keys()), body=body, media_body=MediaFileUpload(video_file_path, chunksize=1024*1024, resumable=True))

    # Upload the video in chunks
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f'Uploaded {int(status.progress() * video_size)} bytes')
    print(f'Video uploaded: {response["id"]}')