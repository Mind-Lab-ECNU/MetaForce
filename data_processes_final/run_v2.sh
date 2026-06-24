# ChartQA pipeline (v2)
python /inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_processes_final/stage1_final/prepare_chartqa_2000_v2.py

python /inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_processes_final/stage2_final/generate_persona_v2.py --mock-persona --split train --input_json /inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_duo_final_v2/ChartQA_2000/train.json
python /inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_processes_final/stage3_final/enhance_chartqa_v2.py --split train

python /inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_processes_final/stage2_final/generate_persona_v2.py --mock-persona --split val --input_json /inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_duo_final_v2/ChartQA_2000/val.json
python /inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_processes_final/stage3_final/enhance_chartqa_v2.py --split val

python /inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_processes_final/stage2_final/generate_persona_v2.py --mock-persona --split test --input_json /inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_duo_final_v2/ChartQA_2000/test.json
python /inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_processes_final/stage3_final/enhance_chartqa_v2.py --split test

# FigureQA pipeline (only train split, v2)
python /inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_processes_final/stage1_final/prepare_figureqa_2000_v2.py

python /inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_processes_final/stage2_final/generate_persona_v2.py --mock-persona --split train --input_json /inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_duo_final_v2/figureqa_2000/train.json --input_parquet /inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_duo_final_v2/figureqa_2000/train.parquet --output_json /inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_duo_final_v2/figureqa_2000/train_with_persona_v2.json --output_parquet /inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_duo_final_v2/figureqa_2000/train_with_persona_v2.parquet
python /inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_processes_final/stage3_final/enhance_figureqa_v2.py --split train

# Stage4_final: split train data 9:1
python /inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_processes_final/stage4_final/split_data_v2.py --split train
python /inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_processes_final/stage4_final/split_data_v2.py --split train --input_dir /inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_duo_final_v2/figureqa_2000 --output_dir /inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_duo_final_v2/figureqa_2000
