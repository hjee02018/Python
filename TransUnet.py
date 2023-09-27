import paddleseg as pdx

# 데이터셋 경로 및 JSON 파일 경로 설정
data_dir = 'dataset'
ㅎㅎjson_file = 'annotations/v1.json'

# 데이터셋 정의
train_dataset = pdx.datasets.Dataset(
    data_dir=data_dir,
    file_list=json_file,
    label_list=["vessel"],  # 라벨 이름 설정
    transforms=pdx.transforms.Compose([
        pdx.transforms.RandomHorizontalFlip(),
        pdx.transforms.Normalize()
    ])
)

# Pre-trained 모델 불러오기
model = pdx.seg.UNet(num_classes=len(train_dataset.labels))

# Fine-tuning을 위한 학습 설정
num_epochs = 10
batch_size = 4
learning_rate = 0.01

optimizer = pdx.optimizer.Momentum(learning_rate=learning_rate, momentum=0.9, params=model.parameters())
scheduler = pdx.optimizer.lr.CosineDecay(learning_rate=learning_rate, cycle_epochs=num_epochs)

# 모델 학습
model.train(
    num_epochs=num_epochs,
    train_dataset=train_dataset,
    train_batch_size=batch_size,
    save_interval_epochs=1,
    log_interval_steps=10,
    save_dir='output',  # 학습 결과 모델 저장 경로
    optimizer=optimizer,
    lr_scheduler=scheduler
)
