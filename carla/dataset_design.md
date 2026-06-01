# Alpamayo追加学習データセット設計案

## 1. 目的

Alpamayoの追加学習に向けて、CARLA上で生成した画像、egomotion、CoC文、ルート情報を一貫した形式で管理する。

今後、CARLA built-in Townだけでなく、OpenStreetMap由来のOpenDRIVEマップや、複数の天候・摩擦係数・運転ポリシーを扱う可能性があるため、拡張しやすいディレクトリ構成とmetadata設計にする。

---

## 2. 基本方針

### 2.1 学習単位は clip とする

Alpamayoの学習データは基本的に20秒クリップ単位で扱う想定のため、学習用データセットは `clips/` 配下に集約する。

* 1 clip = 20秒の画像列 + egomotion + CoC文
* scenario全体の画像は保存しない
* 画像の重複保存を避けるため、`scenarios/` ディレクトリは作らない
* scenario情報は各clipの `metadata.json` に保持する

### 2.2 routeは共通管理する

route情報は複数clipから参照される可能性があるため、`routes/` 配下に共通管理する。

* route.csv
* route_metadata.json
* route_catalog.csv

を用意し、各clipから `route_id` で参照する。

### 2.3 versionは上位フォルダで管理する

dataset versionは上位ディレクトリで分ける。

例：

```text
clips/v001/
routes/v001/
```

そのため、個々のclipフォルダ名には `v001` などのversion文字列は含めない。

ただし、各metadataには `dataset_version` を明記する。

---

## 3. ディレクトリ構成案

```text
alpamayo_ft_data/
  routes/
    v001/
      route_catalog.csv
      builtin/
        Town01/
          sp023/
            rt0001/
              route.csv
              route_metadata.json
            rt0002/
              route.csv
              route_metadata.json
      osm/
        ykhm001/
          sp005/
            rt0001/
              route.csv
              route_metadata.json

  clips/
    v001/
      builtin_Town01_sp023_rt0001_mu040_rainfog_c000/
        metadata.json
        egomotion.csv
        coc.txt
        camera/
          front/
            000000.png
            000001.png
            ...
      builtin_Town01_sp023_rt0001_mu040_rainfog_c001/
        metadata.json
        egomotion.csv
        coc.txt
        camera/
          front/
            000000.png
            000001.png
            ...

  clip_manifest.jsonl
  clip_index.csv
```

---

## 4. ID命名規則

### 4.1 scenario_id

scenario_idは、1つのルート・摩擦係数・天候条件・運転ポリシーで生成された走行条件を表す。

形式：

```text
<map_source>_<map_id>_sp<spawn_id>_<route_id>_mu<mu>_<weather_preset>
```

例：

```text
builtin_Town01_sp023_rt0001_mu040_rainfog
osm_ykhm001_sp005_rt0012_mu035_heavyfog
```

### 4.2 clip_id

clip_idはscenario_idにclip番号を付ける。

形式：

```text
<scenario_id>_c<clip_index>
```

例：

```text
builtin_Town01_sp023_rt0001_mu040_rainfog_c000
builtin_Town01_sp023_rt0001_mu040_rainfog_c001
```

### 4.3 mu表記

摩擦係数は小数を避け、100倍した整数で表す。

```text
0.40 -> mu040
0.35 -> mu035
0.60 -> mu060
```

### 4.4 hashについて

現時点ではファイル名にhashは付けない。

理由：

* 同一version内で同じscenario_id/clip_idを重複生成しない運用にするため
* 既存ディレクトリが存在する場合はエラーにする
* 詳細条件はmetadataで管理するため

将来的に同一条件で複数回生成する必要が出た場合は、hashまたはrun_idの追加を検討する。

---

## 5. route管理

### 5.1 route_id

route_idはrankやdistanceに依存しない安定IDとする。

例：

```text
rt0001
rt0002
rt0003
```

同じmap_id・spawn_idでも、分岐選択や経路形状が異なれば別route_idを割り当てる。

rank、distance、branch_sequenceなどの探索条件はフォルダ名には含めず、`route_metadata.json` に保存する。

---

## 6. route_metadata.json

例：

```json
{
  "dataset_version": "v001",
  "route_id": "rt0001",

  "map": {
    "map_source": "builtin",
    "map_id": "Town01",
    "carla_world_name": "Town01",
    "has_3d_scene": true
  },

  "spawn": {
    "spawn_id": 23,
    "start_transform": {
      "x": 1338.338257,
      "y": -1862.179932,
      "z": 1.0,
      "yaw_deg": 24.485462
    }
  },

  "route_generation": {
    "target_profile": "road_profile_right.csv",
    "distance_m": 220.0,
    "ds_m": 1.0,
    "rank": 2,
    "branch_sequence": [1],
    "max_branches": 1,
    "max_branch_options": 3,
    "script_name": "find_matching_carla_routes.py"
  },

  "route_properties": {
    "route_csv": "route.csv",
    "turn_direction": "right",
    "length_m": 219.7,
    "num_points": 220,
    "max_abs_curvature_1pm": 0.048,
    "mean_abs_curvature_1pm": 0.011
  }
}
```

OSM由来マップの場合は、map情報を以下のようにする。

```json
{
  "map": {
    "map_source": "osm",
    "map_id": "ykhm001",
    "carla_world_name": "OpenDriveWorld",
    "xodr_path": "maps/ykhm001/map.xodr",
    "osm_path": "maps/ykhm001/source.osm",
    "has_3d_scene": false
  }
}
```

---

## 7. clip metadata

各clip配下に `metadata.json` を置く。

例：

```json
{
  "dataset_version": "v001",

  "scenario_id": "builtin_Town01_sp023_rt0001_mu040_rainfog",
  "clip_id": "builtin_Town01_sp023_rt0001_mu040_rainfog_c000",

  "map_source": "builtin",
  "map_id": "Town01",
  "spawn_id": 23,
  "route_id": "rt0001",

  "route": {
    "route_csv": "../../routes/v001/builtin/Town01/sp023/rt0001/route.csv",
    "route_metadata": "../../routes/v001/builtin/Town01/sp023/rt0001/route_metadata.json"
  },

  "road_condition": {
    "friction_mu": 0.4,
    "surface_condition": "wet_low_mu"
  },

  "weather": {
    "weather_preset": "rainfog",
    "cloudiness": 90,
    "precipitation": 90,
    "precipitation_deposits": 80,
    "wetness": 90,
    "fog_density": 70,
    "fog_distance": 20,
    "sun_altitude_angle": 15
  },

  "policy": {
    "policy_version": "safe_curve_policy_v001",
    "v_straight_mps": 8.0,
    "v_turn_mps": 3.0,
    "ax_min_mps2": -1.0,
    "ax_max_mps2": 1.0,
    "jerk_max_mps3": 0.5
  },

  "clip": {
    "clip_index": 0,
    "clip_length_sec": 20.0,
    "clip_start_sec": 8.5,
    "clip_end_sec": 28.5,
    "frame_rate_hz": 10,
    "num_frames": 200
  },

  "labels": {
    "turn_direction": "right",
    "contains_braking": true,
    "contains_turning": true,
    "contains_exit": true,
    "low_visibility": true,
    "low_friction": true
  },

  "outputs": {
    "egomotion": "egomotion.csv",
    "coc": "coc.txt",
    "camera_front": "camera/front"
  }
}
```

---

## 8. egomotion.csv

clip単位の `egomotion.csv` は、clip内で0秒始まりにする。

例：

```csv
t_clip,t_scenario,x_m,y_m,z_m,yaw_rad,vx_mps,ax_mps2,yaw_rate_radps,beta_rad
0.0,8.5,100.0,200.0,1.0,0.10,5.0,-0.2,0.01,0.00
0.1,8.6,100.5,200.0,1.0,0.11,4.9,-0.2,0.01,0.00
```

`t_scenario` を残すことで、元の走行全体における切り出し位置を追跡できるようにする。

---

## 9. clip_manifest.jsonl

AlpamayoのDataset Loaderでは、各clipを1行ずつ記載した `clip_manifest.jsonl` を読む想定にする。

例：

```json
{"clip_id":"builtin_Town01_sp023_rt0001_mu040_rainfog_c000","sample_dir":"clips/v001/builtin_Town01_sp023_rt0001_mu040_rainfog_c000","egomotion":"clips/v001/builtin_Town01_sp023_rt0001_mu040_rainfog_c000/egomotion.csv","coc":"clips/v001/builtin_Town01_sp023_rt0001_mu040_rainfog_c000/coc.txt","camera_front":"clips/v001/builtin_Town01_sp023_rt0001_mu040_rainfog_c000/camera/front","num_frames":200,"split":"train"}
```

---

## 10. clip_index.csv

検索・集計用にCSVも用意する。

例：

```csv
clip_id,scenario_id,map_source,map_id,spawn_id,route_id,friction_mu,weather_preset,clip_index,num_frames,split,valid
builtin_Town01_sp023_rt0001_mu040_rainfog_c000,builtin_Town01_sp023_rt0001_mu040_rainfog,builtin,Town01,23,rt0001,0.40,rainfog,0,200,train,true
builtin_Town01_sp023_rt0001_mu040_rainfog_c001,builtin_Town01_sp023_rt0001_mu040_rainfog,builtin,Town01,23,rt0001,0.40,rainfog,1,200,train,true
```

---

## 11. train / val / test split

splitは物理ディレクトリでは分けず、metadataまたはindexで管理する。

同一route_id由来のclipがtrain/testにまたがると評価が甘くなる可能性があるため、原則としてroute_id単位でsplitする。

例：

```text
rt0001〜rt0080: train
rt0081〜rt0090: val
rt0091〜rt0100: test
```

---

## 12. 今後の拡張

### 12.1 map_source

現時点では以下を想定する。

```text
builtin: CARLA built-in Town
osm: OpenStreetMap由来OpenDRIVE
roadrunner: RoadRunner等で作成したカスタム3Dマップ
```

### 12.2 weather

初期段階ではweather presetを固定する。

例：

```text
rainfog
heavyfog
wet_only
```

ただし、metadataにはpreset名だけでなく、実際のCARLA weather parametersを保存する。

将来的には、presetを中心に以下のような範囲でランダム化する可能性がある。

```text
precipitation: 60〜100
wetness: 70〜100
fog_density: 40〜90
fog_distance: 10〜50
```

### 12.3 hash / run_id

現時点ではclip_idにhashは付けない。
同一clip_idが既に存在する場合は生成エラーにする。

将来的に同一条件で複数runを保存する必要が出た場合は、以下のいずれかを追加する。

```text
run_id
short_hash
created_at
```

---

## 13. まとめ

本設計では、学習用データは `clips/` のみを正とし、画像の重複保存を避ける。
route情報は `routes/` に共通管理し、各clipは `route_id` によって参照する。

ファイル名には以下のみを含める。

```text
map_source, map_id, spawn_id, route_id, mu, weather_preset, clip_index
```

rank、distance、branch_sequence、weatherの詳細パラメータ、policy詳細などはmetadataに保存する。

これにより、CARLA built-in Town、OSM由来マップ、天候preset拡張、摩擦係数バリエーション、複数clip切り出しに対応しやすい構成とする。
