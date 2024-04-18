/* Copyright (C) 2023 Ultralytics
 *   Release under the GPL3.0 Lisence
 *   https://ultralytics.com/license
 */

package jp.gr.java_conf.calpismaker

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.os.Build
import android.util.Log
import com.google.android.gms.tflite.client.TfLiteInitializationOptions
import com.google.android.gms.tflite.gpu.support.TfLiteGpu
import com.google.android.gms.tflite.java.TfLite
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.gpu.GpuDelegateFactory
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel


/*
    CameraXの解析画像(Bitmap)に対し、入力形式に変換後、TensorFlowLiteにて推論を行う
    モデルは main/assets に格納
    推論結果は、objectDetectorListenerを通して、MainFragmentに渡される

    @param context MainFragment
    @param objectDetectorListener ./fragments/MainFragmentのリスナー
 */
class ObjectDetector(
  val context: Context,
  val objectDetectorListener: DetectorListener
) {
    private val TAG = "ObjectDetector"

    // モデルへ入力する際の画像サイズ <- モデル依存
    private val MODEL_SIZE = 640

    // 推論を実行するインタプリタ
    private var interpreter: InterpreterApi? = null
    // Delegateでgpuを使用できるか否か
    private var gpuisavailable = false


    /*
        TensorFlowLiteの初期化
        必ず初期化を完了した上で、推論を実行するように
     */
    init {

        // gpuが対応しているかチェックを行い、実行可能であればそちらを使用
        TfLiteGpu.isGpuDelegateAvailable(context).onSuccessTask { gpuAvailable: Boolean ->
                val optionsBuilder =
                    TfLiteInitializationOptions.builder()
                if (gpuAvailable) {
                    optionsBuilder.setEnableGpuDelegateSupport(true)
                    gpuisavailable = true
                    Log.e(TAG, "gpu is available")
                }
                TfLite.initialize(context, optionsBuilder.build())
                    .addOnSuccessListener {
                        objectDetectorListener.onInitialized()
                    }.addOnFailureListener {
                        objectDetectorListener.onError(
                            "TfLite failed to initialize: "
                                    + it.message
                        )
                    }
            }
    }


    /*
        モデルに解析画像を渡して、推論を実行

        @param image CameraXから取得した解析画像 640*480*3
        @param imageRotation 画像の向き 90/180/270/360度
        @param pitch ジャイロセンサーで取得したデバイスの傾き
     */
    fun detect(image: Bitmap, imageRotation:Int, pitch:Int) {

        // 画像の向きをデバイスに合わせる
        val mat = Matrix()
        mat.postRotate(imageRotation.toFloat())

        val bitmaprotated = Bitmap.createBitmap(
            image,
            0,
            0,
            640,
            480,
            mat,
            true
        )

        // bitmapを正規化済みfloat配列に変換　1*3*640*640　<- モデル依存
        val inputStyle = bitmapToFloatArray(bitmaprotated)

        // 推論実行　-> return [ スコア , マスク ]
        val yoloresults = remBg(inputStyle)

        objectDetectorListener.onResults(
          yoloresults.first,
          yoloresults.second,
          pitch,
          inputStyle)
    }


    // 推論モデルの読み込み
    // @return 読み込んだモデルを返す
    private fun loadMappedFile(): MappedByteBuffer {

        val fileDescriptor = context.assets.openFd("yolov8n-seg-fixed-opt.tflite")
        val modelByteBuffer = fileDescriptor.run {
            val fileChannel = FileInputStream(this.fileDescriptor).channel
            val startOffset = startOffset
            val declaredLength = declaredLength
            fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        }
        return modelByteBuffer
    }


    // インタプリタの初期化
    // Delegate優先順　NNapi -> GPU -> CPU
    fun setInterpreter(){
        try {
            // インタプリタのDelegateオプション
            val interpreterOptions = InterpreterApi.Options()
            var nnApiDelegate: NnApiDelegate? = null

            // NNapi
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                nnApiDelegate = NnApiDelegate()
                interpreterOptions.addDelegate(nnApiDelegate)
                Log.e("Interpreter", "Running on nnApi")
            }
            // gpu -> initで対応を確認
            else if (gpuisavailable){
                interpreterOptions.addDelegateFactory(GpuDelegateFactory())
                Log.e("Interpreter", "Running on gpu")
            }
            // cpu実行 3スレッド
            else {
                interpreterOptions.numThreads = 3
                Log.e("Interpreter", "Running on cpu")
            }

            interpreter = InterpreterApi.create(loadMappedFile(), interpreterOptions)

        }catch (e: Exception) {
        objectDetectorListener.onError(
            "interpreter failed to initialize. See error logs for details"
        )
        Log.e(TAG, "TFLite failed to load model with error: " + e.message)
        }
    }


    // bitmapの各pixelを、0~1に正規化した上でfloat配列に変換
    // 変換の際に左右のパディングも行う right=80 / left=80
    // @param bp CameraX解析用イメージ 640*480*3
    // @return 4次元float配列 1*3*640*640 <- モデル依存
    private fun bitmapToFloatArray(bp: Bitmap): Array<Array<Array<FloatArray>>> {

        val bpcanvas = Bitmap.createBitmap(MODEL_SIZE,MODEL_SIZE, Bitmap.Config.ARGB_8888)
        val can = Canvas(bpcanvas)

        // パディング背景はグレーで埋める
        can.drawARGB(255,114,114,114)
        can.drawBitmap(bp, 80f, 0f, null)

        val intValues = IntArray(MODEL_SIZE * MODEL_SIZE)
        bpcanvas.getPixels(intValues, 0, MODEL_SIZE, 0, 0, MODEL_SIZE, MODEL_SIZE)

        val finalFourDimensionalArray = Array(1) {
            Array(3) {
                Array(MODEL_SIZE) {
                    FloatArray(MODEL_SIZE)
                }
            }
        }

        for (i in 0 until MODEL_SIZE) {
            for (j in 0 until MODEL_SIZE) {
                val pixelValue: Int = intValues[i * MODEL_SIZE + j]
                finalFourDimensionalArray[0][0][i][j] =
                    Color.red(pixelValue).toFloat() / 255f
                finalFourDimensionalArray[0][1][i][j] =
                    Color.green(pixelValue).toFloat() / 255f
                finalFourDimensionalArray[0][2][i][j] =
                    Color.blue(pixelValue).toFloat() / 255f
            }
        }
        return finalFourDimensionalArray
    }


    // 推論の実行を行う
    // @param tsarray 正規化済みfloat配列　解析用イメージ
    // @param return [ スコア, マスク ]
    private fun remBg(tsarray: Array<Array<Array<FloatArray>>>): Pair<Array<Array<FloatArray>>, Array<Array<Array<FloatArray>>>> {

        // スコア格納配列
        val output646 =  Array(1) { Array(38) { FloatArray(8400)}}
        // マスク格納配列
        val output577 =  Array(1) { Array(32) { Array(160) { FloatArray(160)}}}

        // 出力用マップ
        val outputs: MutableMap<Int,
                Any> = HashMap()
        outputs[0] = output646
        outputs[1] = output577

        // tsarrayのままでは入力できないので注意
        val array = arrayOf(tsarray)

        // インタプリタが初期化されていることを確認
        if (interpreter!=null) {

            // 推論実行
            interpreter?.runForMultipleInputsOutputs(array, outputs)
        }
        return Pair(output646,output577)
    }


    interface DetectorListener {
        fun onInitialized()
        fun onError(error: String)
        fun onResults(
            out1: Array<Array<FloatArray>>,
            out2: Array<Array<Array<FloatArray>>>,
            pitch: Int,
            inputstyle: Array<Array<Array<FloatArray>>>
        )
    }
}