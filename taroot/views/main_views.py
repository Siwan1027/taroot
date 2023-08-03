from flask import Blueprint, request, jsonify
from PyKakao import Local
import config
import json as js
import random
import requests
import math
import numpy as np
import pandas as pd
import funcs
import asyncio
import aiohttp
from flask import abort

# Blueprint 객체 생성
bp = Blueprint('main', __name__, url_prefix='/')

# PyKaKao Api 객체 생성
api = Local(service_key='441393e5168fde600bc3153390c2ac25')

# 임시 로컬 데이터셋
master_df = pd.read_csv('https://seoul-taroot.s3.ap-northeast-2.amazonaws.com/map/seoul_bicycle_info.csv', index_col= 0)

# sp, tp 찾기
@bp.route('/search/', methods = ['POST'])
def search_point():
    
    def search_coordinates(query):
        df =  api.search_keyword(query, dataframe=True)
        coors = [df.loc[0,'x'], df.loc[0,'y']]
        return coors

    def find_nearest_station_coordinates(coord, master_df):
        # 각 대여소와 sp,tp 사이의 거리를 계산하여 '거리' 열에 추가
        master_df['거리'] = master_df.apply(lambda row: funcs.leng_cal([float(coord[1]), float(coord[0])], [row['위도'], row['경도']]), axis=1)    # '거리' 열을 기준으로 오름차순으로 정렬하여 가장 가까운 행부터 순서대로 반환
        # 가장 가까운 순으로 5개만
        closest_rows = master_df.sort_values(by='거리').head(5)
        # 저장할 리스트 선언
        nearest_stations_coor = []
        # 리스트에 저장
        for _, row in closest_rows.iterrows():
            nearest_stations_coor.append([row.name,row['경도'], row['위도']])
        print(nearest_stations_coor, closest_rows)
        return nearest_stations_coor

    def find_route(sp_coor, tp_coor, headers):
        body = {"coordinates":[sp_coor, tp_coor]}
        call = requests.post('https://api.openrouteservice.org/v2/directions/cycling-regular/geojson', json=body, headers=headers)
        return call

    def check_available (coors):
        # coors = nearest point coors
        for coor in coors:
            # coor의 인덱스 가져오기
            query = coor[0]
            # print(coor[0])
            # 인덱스 넘버로 요청
            # 4f4b4b657a776f673433506d74786a / {os.getenv("seoul_key")}
            ddareung_info = requests.get(f'http://openapi.seoul.go.kr:8088/{config.seoul_key}/json/bikeList/{query}/{query}')
            print('ddareung_info' , ddareung_info)
            ddareung_info_json = js.loads(ddareung_info.content)

            print('ddareung_info_json', ddareung_info_json)
            if int(ddareung_info_json['rentBikeStatus']['row'][0]['rackTotCnt']) > 0:
                rslt = [coor[1], coor[2]]
                # 잔여 대수 ddareung_info_json['rentBikeStatus']['row'][0]['rackTotCnt']
                # coor[1] : 경도 lng, coor[2] : 위도 ltd
                break
        # 주변 따릉이가 없는 경우
        else : 
            # 이건 프론트에 맞게 수정 필요
            # 다른 위치들을 추천한다던가...
            # status code
            return abort(404)

        return rslt


    def main(sp, tp):
        # front 에서 get
        # sp = contents['startPoint']
        # tp = contents['endPoint']
        # 좌표값 탐색
        sp_coor = search_coordinates(sp)
        tp_coor = search_coordinates(tp)

        # 좌표값 기반 최근접 대여소 탐색
        # db 연동해야해
        nearest_sp_coors = find_nearest_station_coordinates(sp_coor, master_df)
        nearest_tp_coors = find_nearest_station_coordinates(tp_coor, master_df)

        available_sp = check_available(nearest_sp_coors)
        available_tp = check_available(nearest_tp_coors)
        
        return available_sp, available_tp

    sp = request.json['startPoint']
    tp = request.json['endPoint']

    sp_coor, tp_coor = main(sp,tp)
    
    return js.dumps({'body':{'startPoint': sp_coor, 'endPoint' : tp_coor}})

    # if __name__ == "__search_point__":
    #     sp_coor, tp_coor = main()
    #     #rslt = [[coor[1], coor[2]],ddareung_info_json['rentBikeStatus']['row'][0]['rackTotCnt']]
        

# 운동 고도 찾기

async def get_elevation_for_coordinate(session, coordinate, google_maps_key):
    url = f"https://maps.googleapis.com/maps/api/elevation/json?locations={coordinate[1]}%2C{coordinate[0]}&key={google_maps_key}"
    async with session.get(url) as response:
        elevation_info = await response.json()
        return [coordinate[0], coordinate[1], round(elevation_info['results'][0]['elevation'])]

@bp.route('/search/exercise/', methods=['POST'])
async def get_elevation():
    result_dic = {}
    result_list = []
    sp = request.json['startPoint']
    tp = request.json['endPoint']
    matrix, center = funcs.make_matrix(sp, tp)
    headers = {}

    async with aiohttp.ClientSession(headers=headers) as session:
        for quadrant, coordinates in matrix.items():
            if len(coordinates) >= 100:
                random_values = random.sample(coordinates, 25)
            else:
                random_values = coordinates

            tasks = [get_elevation_for_coordinate(session, coordinate, config.google_maps_key) for coordinate in random_values]
            elevation_coors = await asyncio.gather(*tasks)

            # Append the coordinates to the result_list
            result_list.extend(elevation_coors)

    # Sort the result_list based on elevation
    result_list.sort(key=lambda x: x[2])

    # Divide result_list into 3 clusters based on elevation
    n = len(result_list) // 3
    clusters = {
        "level 1": result_list[:n],
        "level 2": result_list[n:2*n],
        "level 3": result_list[2*n:]
    }

    # Find the coordinate with the maximum elevation in each cluster
    max_elevation = {}
    for cluster_name, cluster_coordinates in clusters.items():
        max_elevation_coordinate = max(cluster_coordinates, key=lambda x: x[2])
        max_elevation[cluster_name] = max_elevation_coordinate

    return max_elevation


# 맛집 탐색
@bp.route('/search/restaurant/', methods = ['POST'])
def rt_point():
    # sp, tp 형 (경도, 위도)
    sp = request.json['startPoint']
    tp = request.json['endPoint']
    category = request.json['category']

    mid_latitude = (sp[1] + tp[1]) / 2
    mid_longitude = (sp[0] + tp[0]) / 2

    df = api.search_keyword(query=category, category_group_code='FD6', x=mid_longitude, y=mid_latitude, sort='distance', dataframe=True)

    if not df.empty:
        # 최대 5개의 랜덤 인덱스를 선택 (중복되지 않도록)
        num_results = min(5, len(df))
        random_indexes = random.sample(list(df.index), num_results)

        result = []
        for random_index in random_indexes:
            result.append({
                '이름': df.loc[random_index, 'place_name'],
                '주소': df.loc[random_index, 'address_name'],
                '경도': df.loc[random_index, 'x'],
                '위도': df.loc[random_index, 'y']
            })

        return {'documents': result}
    else:
        return "검색 결과가 없습니다."



# 힐링, 카페 탐색
# 1안 DB에서 찾기
# sp = 출발지, tp = 도착지, purpose = 목적('힐링','식사'), category = 대분류 중 1
@bp.route('/search/healing/', methods = ['POST'])
def waypoint():

    sp = request.json['startPoint']
    tp = request.json['endPoint']
    category = request.json['category']

    # 차후 DB 연동 후 실행
    target_datas = pd.read_csv('https://seoul-taroot.s3.ap-northeast-2.amazonaws.com/map/healing_points.csv', index_col=0)
    target_datas = target_datas[target_datas['대분류'] == category]
    
    mid_point = [round((sp[0] + tp[0]) / 2, 6), round((sp[1] + tp[1]) / 2, 6)]

    # 각 레스토랑과 mid_point 사이의 거리를 계산하여 '거리' 열에 추가
    target_datas['거리'] = target_datas.apply(lambda row: funcs.leng_cal(mid_point, [row['위도'], row['경도']]), axis=1)

    # '거리' 열을 기준으로 오름차순으로 정렬하여 가장 가까운 행부터 순서대로 반환
    closest_rows = target_datas.sort_values(by='거리').head(3)

    # 상호명, 주소, 위도, 경도를 추출하여 원하는 형태의 딕셔너리로 반환
    result = {}
    for i, row in enumerate(closest_rows.itertuples(), 1):
        result[f"{i}번째"] = {
            '상호명': row.상호명,
            '주소': row.주소,
            '위도': row.위도,
            '경도': row.경도,
            '중간지점' : mid_point
        }

    return result