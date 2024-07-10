# Bili-Search

Search videos, danmakus and replies in Bilibili.

## Data preprocessing

Audio:

- Extract audios from videos
- Speech recognition from audios

Vision:

- Extract frames from videos
- Label and describe frames

Statistics:

- Author:
  - mid, name, pub_location

- Video:
  - bvid, aid, cid, tid, tname
  - title, pubdate, duraiton, description, pic, first_frame, 
  - view, danmaku, reply, favorite, coin, share, like

## Commands

```sh
export BILI_MID=<mid>

cd ~/repos/bili-scraper
python -m workers.user_worker -p -v -d -od -m $BILI_MID

cd ~/repos/bili-search
python -m converters.video_to_audio -m $BILI_MID -o
python -m converters.audio_to_subtitle -m $BILI_MID -o
python -m elastics.video_details_indexer -m $BILI_MID
```