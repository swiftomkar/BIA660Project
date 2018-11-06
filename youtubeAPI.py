import httplib2
import os
import sys
import csv
import json
from apiclient.discovery import build_from_document
from apiclient.errors import HttpError
from oauth2client.client import flow_from_clientsecrets
from oauth2client.file import Storage
from oauth2client.tools import argparser, run_flow

CLIENT_SECRETS_FILE = "client_secrets.json"
YOUTUBE_READ_WRITE_SSL_SCOPE = "https://www.googleapis.com/auth/youtube.force-ssl"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
MISSING_CLIENT_SECRETS_MESSAGE = "WARNING: Please configure OAuth 2.0"


# Authorize the request and store authorization credentials.
def get_authenticated_service(args):
    flow = flow_from_clientsecrets(CLIENT_SECRETS_FILE, scope=YOUTUBE_READ_WRITE_SSL_SCOPE,
                                   message=MISSING_CLIENT_SECRETS_MESSAGE)
    storage = Storage("%s-oauth2.json" % sys.argv[0])
    credentials = storage.get()

    if credentials is None or credentials.invalid:
        credentials = run_flow(flow, storage, args)

    with open("youtube-v3-discoverydocument.json", "r",encoding='utf-8') as discovery:
        doc = discovery.read()

    return build_from_document(doc, http=credentials.authorize(httplib2.Http()))


# Call the API's commentThreads.list method to list the existing comments.
def get_comments(youtube, video_id, channel_id, nextpg):
    currVideoId=video_id
    results = youtube.commentThreads().list(
        part="snippet,replies",
        videoId=video_id,
        channelId=channel_id,
        textFormat="plainText",
        maxResults=100,
        pageToken=nextpg
    ).execute()
    f_name=video_id+'Comments.csv'
    file=open(f_name,'a')
    print('.')
    try:
        nextpg=results['nextPageToken']
    except:
        nextpg=''
    writer = csv.writer(file)
    writer.writerow(["id", "created_at", "user_id", "screen_name", "text"])
    for item in results["items"]:
        comment = item["snippet"]["topLevelComment"]
        author = json.dumps(comment["snippet"]["authorDisplayName"])
        text = json.dumps(comment["snippet"]["textDisplay"])
        timestamp=json.dumps(comment["snippet"]["publishedAt"])
        writer.writerow(['N/A',timestamp,author,'N/A',text])
    file.close()
    if nextpg is not '':
        get_comments(youtube, currVideoId, None, nextpg)
    return



if __name__ == "__main__":
  # The "videoid" option specifies the YouTube video ID that uniquely
  # identifies the video for which the comments will be fetched
  argparser.add_argument("--videoid",
    help="Required; ID for video for which the comment will be inserted.")
  args = argparser.parse_args()

  if not args.videoid:
    exit("Please specify videoid using the --videoid= parameter.")

  youtube = get_authenticated_service(args)
  try:
    nextID = get_comments(youtube, args.videoid, None,'')
  except HttpError as e:
    print ("An HTTP error %d occurred:\n%s" % (e.resp.status, e.content))
  else:
    print ("Fetched and stored all comments for given videoid")
