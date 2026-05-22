import tensorflow as tf

print("1. TensorFlow版のResNet50をダウンロード中...")
# PyTorchと同じ「ResNet50（ImageNet学習済み）」を直接TensorFlowから呼び出します
model = tf.keras.applications.ResNet50(weights='imagenet')

print("2. TFLite形式に変換中...")
# ONNXを経由せず、直接TFLiteに変換できるためエラーが起きません
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# スマホ向けに最適化（少し精度を落としてファイルサイズを劇的に軽くする）
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

tflite_path = "resnet50.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"大成功！ {tflite_path} が無事に生成されました。")