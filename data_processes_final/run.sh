# ChartQA pipeline
python /inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_processes_final/stage1/prepare_chartqa_2000.py

python /inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_processes_final/stage2/generate_persona.py --split train --input_json /inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_duo_final/ChartQA_2000/train.json
python /inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_processes_final/stage3/enhance_chartqa.py --split train

python /inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_processes_final/stage2/generate_persona.py --split val --input_json /inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_duo_final/ChartQA_2000/val.json
python /inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_processes_final/stage3/enhance_chartqa.py --split val

python /inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_processes_final/stage2/generate_persona.py --split test --input_json /inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_duo_final/ChartQA_2000/test.json
python /inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_processes_final/stage3/enhance_chartqa.py --split test

# FigureQA pipeline (only train split)
python /inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_processes_final/stage1/prepare_figureqa_2000.py

python /inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_processes_final/stage2/generate_persona.py --split train --input_json /inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_duo_final/figureqa_2000/train.json --input_parquet /inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_duo_final/figureqa_2000/train.parquet --output_json /inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_duo_final/figureqa_2000/train_with_persona.json --output_parquet /inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_duo_final/figureqa_2000/train_with_persona.parquet
python /inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_processes_final/stage3/enhance_figureqa.py --split train

# Stage4: split train data 9:1
python /inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_processes_final/stage4/split_data.py --split train
python /inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_processes_final/stage4/split_data.py --split train --input_dir /inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_duo_final/figureqa_2000 --output_dir /inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/verl_m/data_duo_final/figureqa_2000 