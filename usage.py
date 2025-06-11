from main_api import VietnameseToneAPI

api = VietnameseToneAPI()
api.load_model()
while True:
    intput_text = input("Nhập văn bản: ")
    if intput_text == "q":
        break
    results = api.restore_tones(intput_text, max_results=10)
    print(results)
    print("-"*100)