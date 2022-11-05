import * as tf from "@tensorflow/tfjs"
export async function ImageDataToTensor(input: ImageData):
    Promise<tf.Tensor3D> {
  // const rawImageData = tf.util.encodeString(base64, 'base64');
  // const TO_UINT8ARRAY = true;
  // const {width, height, data} = jpeg.decode(rawImageData, TO_UINT8ARRAY);
  // Drop the alpha channel info
  const {width, height, data} = input;




  const buffer = new Uint8Array(width * height * 3);
  let offset = 0;  // offset into original data
  for (let i = 0; i < buffer.length; i += 3) {
    buffer[i] = data[offset]!;
    buffer[i + 1] = data[offset + 1]!;
    buffer[i + 2] = data[offset + 2]!;

    offset += 4;
  }
  return tf.tensor3d(buffer, [height, width, 3], "float32");
}