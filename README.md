# Запуск
Тестировалось под:
- Ubuntu 18.04
- Cuda 10.0
- Python 3.6

```sh
pip3 install -r requirements.txt
sudo chmod +x get_nn.sh
./get_nn.sh
python3 test.py --input INPUT_FLD --output OUTPUT_FLD
```

В случае проблем с версией Cuda можно запускать с флагом `--force_cpu`
```sh
python3 test.py --input INPUT_FLD --output OUTPUT_FLD --force_cpu
```