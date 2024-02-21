from PIL import Image
import numpy as np

def show_filters(layers):
    for layer in layers:
        if hasattr(layer, 'w'):
            if layer.w.ndim > 3:
                for n in range(layer.w.shape[0]):
                    filters_combined = []

                    for c in range(layer.w.shape[1]):
                        filter = layer.w[n, c, :, :]

                        min_val = np.min(filter)
                        max_val = np.max(filter)

                        filter_scaled = ((filter - min_val) / (max_val - min_val) * 255).astype(np.uint8)

                        filters_combined.append(filter_scaled)

                        if c < layer.w.shape[1] - 1:
                            filters_combined.append(np.zeros((filter_scaled.shape[0], round(filter_scaled.shape[1] / 2)), dtype=np.uint8))

                    combined_image = np.hstack(filters_combined)
                    combined_image = Image.fromarray(combined_image)
                    combined_image.show()

'''
# 画像の読み込み
image = Image.open("/Users/yudaihamashima/products/networks/image/617.jpg")  # 画像のファイル名を適切なものに置き換えてください

# 画像をRGB輝度値の行列に変換
rgb_array = np.array(image)

# 結果の表示
print(rgb_array.shape)  # 行列の形状を表示
#print(rgb_array)  # RGB輝度値の行列を表示

rgb_array = np.random.randint(0, 256, size=(3, 100, 100), dtype=np.uint8)

# RGB輝度値の行列から画像を作成
image = Image.fromarray(np.transpose(rgb_array, (1, 2, 0)))

# 画像を保存または表示する場合
# image.save("output_image.jpg")  # 画像を保存する場合
image.show()
'''