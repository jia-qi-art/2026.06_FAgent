# 宸ヤ笟鏃跺簭寮傚父璇婃柇 Agent 骞冲彴锛圧elation-EVGAT Demo锛?
杩欐槸涓€涓嫭绔嬬増 MVP锛氶」鐩唴鍖呭惈 Relation-EVGAT 蹇呰绠楁硶鑴氭湰銆乄aDI/SMD 鏍蜂緥鏁版嵁鍜岄粯璁?outputs锛屼笉闇€瑕佽繍琛屾椂寮曠敤鏃ч」鐩洰褰曘€?
## 鍚姩鍚庣

```powershell
python -m pip install -r backend\requirements.txt
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

## 鍚姩鍓嶇

```powershell
cd frontend
npm install
npm run dev
```

鎵撳紑 `http://127.0.0.1:5173/dashboard`銆?
## 涓昏鍔熻兘

- `/dashboard`锛氬疄鏃剁洃鎺с€佸紓甯稿垎鏁般€佹姤璀︽椂闂寸嚎銆丄gent 闈㈡澘銆?- `/relations`锛氫紶鎰熷櫒鍏崇郴鍥俱€乀op 閫€鍖栬竟銆佽竟鍚戦噺瀵规瘮銆?- `/root-cause`锛歍op-K 鏍瑰洜鍊欓€夈€佽瘉鎹崱鐗囥€佸巻鍙叉洸绾裤€?- `/report`锛氳鍒?Agent 璇婃柇瀵硅瘽銆佹姤鍛婄敓鎴愩€佸伐鍏疯皟鐢ㄦ棩蹇椼€?
## 鍚庣鎺ュ彛

- `GET /api/health`
- `GET /api/datasets`
- `POST /api/jobs/train`
- `GET /api/jobs/{job_id}`
- `GET /api/overview?dataset=WaDI_A2_ds10`
- `GET /api/timeseries?dataset=WaDI_A2_ds10`
- `GET /api/relation-graph?dataset=WaDI_A2_ds10&event_id=1`
- `GET /api/root-cause?dataset=WaDI_A2_ds10&event_id=1`
- `POST /api/agent/ask`
- `GET /api/report?dataset=WaDI_A2_ds10&event_id=1`

## 璁粌璇存槑

鍓嶇鈥滃惎鍔ㄨ交閲忚缁冣€濅細璋冪敤鏈」鐩唴鐨?`relation_evgat/run_top_ready_relation_gat.py`锛岄粯璁や娇鐢?`epochs=1`銆乣max_train_windows=1000` 鍋氬揩閫熼棴鐜獙璇併€傞娆℃紨绀烘棤闇€绛夊緟璁粌锛岀郴缁熶細鐩存帴璇诲彇澶嶅埗杩涙潵鐨?`outputs/top_ready_relation_gat/WaDI_A2_ds10/full_joint`銆?
