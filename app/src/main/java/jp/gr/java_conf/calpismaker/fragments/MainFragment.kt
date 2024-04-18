/*
 * Copyright (c) 2017-2022 Chaquo Ltd and contributors

 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/*
 * Copyright (c) 2014 daimajia
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*
 * Copyright (c) 2019 Jorge Garrido Oval <firezenk@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

package jp.gr.java_conf.calpismaker.fragments

import android.annotation.SuppressLint
import android.content.Context.SENSOR_SERVICE
import android.content.res.Configuration
import android.graphics.*
import android.graphics.drawable.AnimationDrawable
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.*
import android.view.GestureDetector.SimpleOnGestureListener
import android.view.ViewGroup.MarginLayoutParams
import android.view.animation.Animation
import android.view.animation.AnimationUtils
import android.widget.*
import android.widget.SeekBar.OnSeekBarChangeListener
import androidx.annotation.RequiresApi
import androidx.camera.core.*
import androidx.camera.core.Camera
import androidx.camera.core.ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.core.view.isInvisible
import androidx.core.widget.NestedScrollView
import androidx.fragment.app.Fragment
import androidx.navigation.Navigation
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.daimajia.androidanimations.library.Techniques
import com.daimajia.androidanimations.library.YoYo
import com.google.android.material.bottomsheet.BottomSheetBehavior
import jp.gr.java_conf.calpismaker.ObjectDetector
import jp.gr.java_conf.calpismaker.R
import jp.gr.java_conf.calpismaker.databinding.FragmentMainBinding
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.ThreadFactory
import kotlin.math.*
import kotlin.random.Random


/*
    MainFragment

    Android CameraX からイメージを取得し、ObjectDetector に Bitmap で渡す
    objectDetectorListener.onResults で受け取った推論結果を、Pythonモジュール に渡す
    完成したイメージを Pythonモジュールから bytearray で受け取り、Bitmap に変換後 calpismask に格納
    ( handler で calpismask の bitmap を周期的にディスプレイに描写 )

 */
class MainFragment : Fragment(), ObjectDetector.DetectorListener, SensorEventListener{

    private val TAG = "MainFragment"

    private var _fragmentMainBinding: FragmentMainBinding? = null
    private val fragmentMainBinding
        get() = _fragmentMainBinding!!

    // ボトムシートの挙動を管理
    private lateinit var bottombehavior: BottomSheetBehavior<View>

    // 推論実行オブジェクト -> ObjectDetector
    private lateinit var objectDetector: ObjectDetector

    // Android CameraX
    // https://developer.android.com/media/camera/camerax/preview?hl=ja
    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null

    // FloatingIceのAnimationDrawable
    private lateinit var overflowanime: AnimationDrawable

    // エラー表示のAnimationDrawable
    private lateinit var erroranime: AnimationDrawable

    // スマホ画面のサイズ
    private var layoutWidth: Int? = null
    private var layoutHeight: Int? = null

    // ディスプレイに表示するカルピスイメージを保持する
    private var calpismask:Bitmap?=null
    private lateinit var bitmapBuffer: Bitmap

    // ボトムシートが展開された際に、気泡をバックグラウンドで描写するためのhandler
    private val handler : Handler = Handler(Looper.getMainLooper())

    // カルピスイメージ(Bitmap)をバックグラウンドで常に描写するためのhandler
    private val handler2 : Handler = Handler(Looper.getMainLooper())

    // CameraXのExecutor
    private lateinit var cameraExecutor: ExecutorService

    // PythonモジュールのExecutor
    private lateinit var calpisExecutor: ExecutorService

    // ジャイロセンサー
    private lateinit var mSensorManager:SensorManager
    private var mIsMagSensor = false
    private var mIsAccSensor = false
    private var mMagneticValues: FloatArray? = null
    private var mAccelerometerValues: FloatArray? = null
    // ピッチ (X軸回転)
    private var mPitchX = 0
    // ロール (Y軸回転)
    private var mRollY = 0
    // 方位角 (Z軸回転)
    private var mAzimuthZ = 0
    private val rad2deg = 180 / PI
    private val matrixSIZE = 16
    private val dimension = 3

    // Pythonモジュールで推論実行が可能 -> true
    // Pythonモジュールで推論実行中 -> false
    private var pyIsRunnable = true

    // CameraXのプレビューが表示されているか否か
    private var hassurfaceprovider = false

    // 停止ボタンが押されていない -> true
    private var nonfreeze = true

    // カルピスイメージの表示時間カウンター
    private var holdcount = 0

    // カルピスイメージを表示するイメージビュー
    private lateinit var maskview:ImageView

    // カルピスイメージビューのデバイス画面上でのX座標
    private var contentsx = 0

    // カルピスイメージビューのデバイス画面上でのY座標
    private var contentsy = 0

    // ヘルプ表示のオンオフ切り替え
    private var helpon=false


    // Pythonモジュール用のパラメータ -----------------------------

    // 動作モード切替設定値 希釈モード -> 1 / 目安モード -> 2
    private var execMode = 1

    // カルピス容量設定値 1から10のInt
    private var glassVolume = 7

    // カルピス濃度設定値 1から10のInt
    private var concentration = 5

    // 厚底グラス設定値 true -> 1 / false -> 0
    private var thickBottom = 0

    // -------------------------------------------------------



    override fun onInitialized() {

        // インタプリタの初期化と推論モデルの読み込み
        objectDetector.setInterpreter()

        // CameraX実行用の単一スレッドプールを作成
        cameraExecutor = Executors.newSingleThreadExecutor()

        // Pythonモジュール実行用の単一スレッドプールを作成　-> 優先度高
        calpisExecutor = Executors.newSingleThreadExecutor(makeThreadFactory(Thread.MAX_PRIORITY))

        // CameraXプレビュー画面の作成後、CameraXの設定を行う
        fragmentMainBinding.viewFinder.post {
            setUpCamera()
        }

        // プログレスバーの非表示
        fragmentMainBinding.progressCircular.visibility = View.GONE
    }


    // スレッドの優先度を作成
    // @param priority スレッドの優先度 Thread.MAX_PRIORITY
    private fun makeThreadFactory(priority: Int): ThreadFactory {
        return ThreadFactory { r ->
            val thread = Thread(r)
            thread.priority = priority
            thread
        }
    }


    override fun onResume() {
        super.onResume()

        // カメラのpermissionがない場合に、PermissionsFragmentに遷移
        if (!PermissionsFragment.hasPermissions(requireContext())) {
            Navigation.findNavController(requireActivity(), R.id.fragment_container)
                .navigate(MainFragmentDirections.actionMainToPermissions())
        }

        // 加速度センサー・磁気センサーが有効か確認し、リスナーリストに登録
        val sensors: List<Sensor> = mSensorManager.getSensorList(Sensor.TYPE_ALL)
        for (sensor in sensors) {
            if (sensor.type == Sensor.TYPE_MAGNETIC_FIELD) {
                mSensorManager.registerListener(
                    this,
                    sensor,
                    SensorManager.SENSOR_DELAY_GAME
                )
                mIsMagSensor = true
            }
            if (sensor.type == Sensor.TYPE_ACCELEROMETER) {
                mSensorManager.registerListener(
                    this,
                    sensor,
                    SensorManager.SENSOR_DELAY_GAME
                )
                mIsAccSensor = true
            }
        }

        // 前回に作成したカルピスイメージを削除
        calpismask = null
    }


    @SuppressLint("MissingPermission")
    override fun onCreateView(
      inflater: LayoutInflater,
      container: ViewGroup?,
      savedInstanceState: Bundle?
    ): View {
        _fragmentMainBinding = FragmentMainBinding.inflate(inflater, container, false)

        // センサーマネージャの取得
        mSensorManager = requireActivity().getSystemService(SENSOR_SERVICE) as SensorManager

        return fragmentMainBinding.root
    }


    @RequiresApi(Build.VERSION_CODES.R)
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        objectDetector = ObjectDetector(
            context = requireContext(),
            objectDetectorListener = this)

        // ボトムシートの挙動を取得
        val bottomid = activity?.findViewById<View>(R.id.bottom_sheet) as NestedScrollView
        bottombehavior = BottomSheetBehavior.from(bottomid)

        // FloatingIceアニメーション開始
        setYoYo()

        // 各種ボタン、スライダーのリスナーの設定
        initListeners()

        // Chaquopyモジュール取得
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(requireContext()))
        }

        // レイアウトが完了してから画面サイズを取得
        view.post(Runnable {
            layoutWidth = view.width
            layoutHeight = view.height
        })

        // 画面左上アイコンにAnimationDrawableの設定
        activity?.findViewById<ImageView>(R.id.overflow_icon)?.apply {
            setBackgroundResource(R.drawable.overflow)
        }
        overflowanime = activity?.findViewById<ImageView>(R.id.overflow_icon)?.background as AnimationDrawable

        activity?.findViewById<ImageView>(R.id.error_icon)?.apply {
            setBackgroundResource(R.drawable.error_anime)
        }
        erroranime = activity?.findViewById<ImageView>(R.id.error_icon)?.background as AnimationDrawable


        // カルピスイメージの描写をhandlerで周期的に実行
        // calpismask の bitmap を更新することで、イメージの切り替えを行う
        maskview = activity?.findViewById(R.id.mask)!!
        val r: Runnable = object : Runnable {
            override fun run() {

                // ディスプレイ上での表示座標を指定
                val viewMargin = maskview.layoutParams as MarginLayoutParams
                viewMargin.topMargin = contentsy
                viewMargin.leftMargin = contentsx
                maskview.layoutParams = viewMargin

                // カメラプレビューが表示されており、停止ボタンが押されていない場合に限り、イメージを更新
                if (hassurfaceprovider && nonfreeze) {

                    // nullの場合は、非表示
                    if (calpismask == null) {
                        maskview.visibility = View.INVISIBLE
                    }
                    // 非表示の場合は表示切替
                    else if (maskview.isInvisible) {
                        maskview.visibility = View.VISIBLE
                        maskview.setImageBitmap(calpismask)
                    }
                    else {
                        maskview.setImageBitmap(calpismask)
                    }
                }
                handler2.postDelayed(this, 50)
            }
        }
        // 実行
        handler2.post(r)
    }


    override fun onDestroyView() {

        _fragmentMainBinding = null
        super.onDestroyView()

        // スレッドプールのシャットダウン
        cameraExecutor.shutdown()
        calpisExecutor.shutdown()

        // スレッドで実行予定だったタスクを全て削除
        handler.removeCallbacksAndMessages(null)
        handler2.removeCallbacksAndMessages(null)
    }


    // 画面停止ボタンが押された際の挙動を設定
    private val mOnGestureListener: SimpleOnGestureListener = object : SimpleOnGestureListener() {

        // 画面停止ボタンがシングルタップされた際の挙動
        override fun onSingleTapConfirmed(e: MotionEvent): Boolean {

            // CameraXプレビューを持っていない場合は無視
            if (hassurfaceprovider) {

                // 停止ボタンが押されていない際の挙動
                if (nonfreeze) {
                    fragmentMainBinding.freeze.setImageResource(
                        resources.getIdentifier(
                            "lock_calpis", "drawable",
                            activity?.packageName
                        )
                    )
                }
                // 停止ボタンが押されている際の挙動
                else {
                    fragmentMainBinding.freeze.setImageResource(
                        resources.getIdentifier(
                            "stop_button", "drawable",
                            activity?.packageName
                        )
                    )
                }
                nonfreeze = !nonfreeze
            }
            return super.onSingleTapConfirmed(e)
        }

        // 画面停止ボタンがダブルタップされた際の挙動
        override fun onDoubleTap(e: MotionEvent): Boolean {

            if (hassurfaceprovider) {
                preview?.setSurfaceProvider(null)
                fragmentMainBinding.freeze.setImageResource(resources.getIdentifier("lock_camera", "drawable",
                    activity?.packageName))
            }
            else {
                preview?.setSurfaceProvider(fragmentMainBinding.viewFinder.surfaceProvider)
                fragmentMainBinding.freeze.setImageResource(resources.getIdentifier("stop_button", "drawable",
                    activity?.packageName))
                nonfreeze = true
            }
            // SurfaceProviderを解除　
            // hassurfaceproviderをfalseにすることで、CameraXプレビューの更新を停止
            hassurfaceprovider = ! hassurfaceprovider

            return super.onDoubleTap(e)
        }
    }


    // UI要素に対するリスナーの設定
    private fun initListeners() {

        // 画面停止ボタンのシングルタップ/ダブルタップを検出
        val doubletopdetect = GestureDetector(context,mOnGestureListener)
        fragmentMainBinding.freeze.setOnTouchListener { v, event ->
            doubletopdetect.onTouchEvent(event)
        }

        // ヘルプボタン
        fragmentMainBinding.helpbutton.setOnClickListener {
            // 既にヘルプが表示されている場合
            if (helpon) {
                val animationout: Animation = AnimationUtils.loadAnimation(
                    context,
                    R.anim.alpha_fadeout
                )
                fragmentMainBinding.helpview.startAnimation(animationout)
                fragmentMainBinding.helpclose.startAnimation(animationout)
                helpon = !helpon
            }
            else {
                val animationin: Animation = AnimationUtils.loadAnimation(
                    context,
                    R.anim.alpha_fadein
                )
                fragmentMainBinding.helpview.startAnimation(animationin)
                fragmentMainBinding.helpclose.startAnimation(animationin)
                helpon = !helpon
            }
        }

        // ヘルプを閉じるボタン
        fragmentMainBinding.helpclose.setOnClickListener {
            if (helpon) {

                val animationout: Animation = AnimationUtils.loadAnimation(
                    context,
                    R.anim.alpha_fadeout
                )
                fragmentMainBinding.helpview.startAnimation(animationout)
                fragmentMainBinding.helpclose.startAnimation(animationout)
                helpon = !helpon
            }
        }

        // シークバー（濃度）
        fragmentMainBinding.bottomSheet.concentrationbar.setOnSeekBarChangeListener(object : OnSeekBarChangeListener {

            // @param seekBar シークバーオブジェクト(濃度)
            // @param i 現在の進捗度
            // @param b ユーザーによる変更ならばtrue
            override fun onProgressChanged(seekBar: SeekBar, i: Int, b: Boolean) {

                // 濃度は、progress＋１となる点に注意
                concentration = fragmentMainBinding.bottomSheet.concentrationbar.progress + 1

                // 表示テキストを連動して変更
                fragmentMainBinding.bottomSheet.multi.text = concentration.toString()

                // 濃さに応じて、ハンドルアイコンを変更
                val imagename =  resources.getIdentifier("kosa${concentration}", "drawable",activity?.packageName)
                fragmentMainBinding.bottomSheet.concentrationbar.thumb = resources.getDrawable(imagename)
            }
            override fun onStartTrackingTouch(seekBar: SeekBar) {}
            override fun onStopTrackingTouch(seekBar: SeekBar) {}
        })

        // シークバー（容量）
        fragmentMainBinding.bottomSheet.concentrationbar2.setOnSeekBarChangeListener(object : OnSeekBarChangeListener {

            // @param seekBar シークバーオブジェクト(容量)
            // @param i 現在の進捗度
            // @param b ユーザーによる変更ならばtrue
            override fun onProgressChanged(seekBar: SeekBar, i: Int, b: Boolean) {

                // 容量は、progress＋１となる点に注意
                glassVolume = fragmentMainBinding.bottomSheet.concentrationbar2.progress + 1

                // 表示テキストを連動して変更
                fragmentMainBinding.bottomSheet.multi2.text = glassVolume.toString()

                // 容量に応じて、ハンドルアイコンを変更
                val imagename2 =  resources.getIdentifier("l${glassVolume}", "drawable",activity?.packageName)
                fragmentMainBinding.bottomSheet.concentrationbar2.thumb = resources.getDrawable(imagename2)
            }
            override fun onStartTrackingTouch(seekBar: SeekBar) {}
            override fun onStopTrackingTouch(seekBar: SeekBar) {}
        })

        // モード切替スイッチ　
        // change_mode が 1 なら希釈モード、　2 なら目安モード
        fragmentMainBinding.modeSwitch.setOnCheckedChangeListener(CompoundButton.OnCheckedChangeListener { buttonView, isChecked ->

            // 希釈モード -> 目安モード　のモード切替の場合
            if (isChecked) {
                execMode = 1

                // ハンドルの変更
                fragmentMainBinding.execMode.setImageResource(resources.getIdentifier("switchicon1", "drawable",
                    activity?.packageName))
                // 背景の変更
                fragmentMainBinding.bottomSheet.wallpaper.setImageResource(resources.getIdentifier("background1_2", "drawable",
                    activity?.packageName))
                // アイコンの変更
                fragmentMainBinding.modeSwitch.setThumbResource(resources.getIdentifier("switch_fill", "drawable",
                    activity?.packageName))
                // ボトムシートの以下項目を非表示に変更
                // 厚底チェックボックス、倍率シークバー、濃度シークバー、シークバー背景
                fragmentMainBinding.bottomSheet.checkBox.visibility = View.INVISIBLE
                fragmentMainBinding.bottomSheet.multi2.visibility = View.INVISIBLE
                fragmentMainBinding.bottomSheet.concentrationbar2.visibility = View.INVISIBLE
                fragmentMainBinding.bottomSheet.seekbarBack2.visibility = View.INVISIBLE

            }
            // 目安モード -> 希釈モード　のモード切替の場合
            else {
                execMode = 2

                // ハンドルの変更
                fragmentMainBinding.execMode.setImageResource(resources.getIdentifier("switchicon2", "drawable",
                    activity?.packageName))
                // 背景の変更
                fragmentMainBinding.bottomSheet.wallpaper.setImageResource(resources.getIdentifier("background2_2", "drawable",
                    activity?.packageName))
                // アイコンの変更
                fragmentMainBinding.modeSwitch.setThumbResource(resources.getIdentifier("switch_empty", "drawable",
                    activity?.packageName))
                // ボトムシートの以下項目を表示に変更
                // 厚底チェックボックス、倍率シークバー、濃度シークバー、シークバー背景
                fragmentMainBinding.bottomSheet.checkBox.visibility = View.VISIBLE
                fragmentMainBinding.bottomSheet.multi2.visibility = View.VISIBLE
                fragmentMainBinding.bottomSheet.concentrationbar2.visibility = View.VISIBLE
                fragmentMainBinding.bottomSheet.seekbarBack2.visibility = View.VISIBLE
            }
        })

        // 厚底チェックボックス
        fragmentMainBinding.bottomSheet.checkBox.setOnClickListener {
            val check: Boolean = fragmentMainBinding.bottomSheet.checkBox.isChecked
            if (check) {
                thickBottom = 1 // 厚底
            } else {
                thickBottom = 0 // 厚底でない
            }
        }

        // ボトムシートビューの挙動を取得
        // シート全体が表示された時点でBubbleEmitterをHandlerで実行
        bottombehavior.addBottomSheetCallback(object : BottomSheetBehavior.BottomSheetCallback() {

            // @param bottomSheet シークバーやボタンなど設定項目が入ったボトムシート
            // @param newState ボトムシートの状態を表す
            override fun onStateChanged(bottomSheet: View, newState: Int) {
                // シート全体が表示されたとき
                if (newState==BottomSheetBehavior.STATE_EXPANDED){
                    val r2: Runnable = object : Runnable {
                        override fun run() {
                            val size = Random.nextInt(20, 80)
                            fragmentMainBinding.bottomSheet.bubbleEmitter.setColors(fill = resources.getColor(R.color.bubble)
                                , stroke = resources.getColor(R.color.stroke))
                            fragmentMainBinding.bottomSheet.bubbleEmitter.emitBubble(size)

                            handler.postDelayed(this, Random.nextLong(60, 300))
                        }
                    }
                    handler.postDelayed(r2, Random.nextLong(100, 500))
                }
                // シートの一部が隠れているので、BubbleEmitterを停止
                else{handler.removeCallbacksAndMessages(null)}
            }
            override fun onSlide(bottomSheet: View, slideOffset: Float) {}
        })
    }


    // CameraXを利用するためのインスタンスを取得
    private fun setUpCamera() {

        val cameraProviderFuture = ProcessCameraProvider.getInstance(requireContext())

        cameraProviderFuture.addListener(
            {
                cameraProvider = cameraProviderFuture.get()
                bindCameraUseCases()
            },
            // メインスレッドで実行
            ContextCompat.getMainExecutor(requireContext())
        )
    }


    // CameraXのユースケース（プレビュー・画像解析）をProcessCameraProviderにバインド
    @SuppressLint("UnsafeOptInUsageError")
    private fun bindCameraUseCases() {

        val cameraProvider =
            cameraProvider ?: throw IllegalStateException("Camera initialization failed.")

        // 背面カメラの使用を要求
        val cameraSelector =
            CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_BACK).build()

        // アスペクト比 4:3
        // 角度はディスプレイの向きに合わせる
        preview =
            Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(fragmentMainBinding.viewFinder.display.rotation)
                .build()

        // アスペクト比 4:3
        // 角度はディスプレイの向きに合わせる
        // バッファリングは最新の画像のみ
        // 出力形式は RGBA 32 ビットカラー
        imageAnalyzer =
            ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(fragmentMainBinding.viewFinder.display.rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { image ->
                        if (!::bitmapBuffer.isInitialized) {
                            bitmapBuffer = Bitmap.createBitmap(
                                image.width,
                                image.height,

                                // ARGB_8888である点に注意
                                Bitmap.Config.ARGB_8888
                            )
                        }
                        detectObjects(image, mPitchX)
                    }
                }

        // 既存のユースケースのバインド解除
        cameraProvider.unbindAll()

        try {
            // ユースケースをフラグメントのライフサイクル にバインド
            // -> フラグメントがアクティブな間だけカメラを使用する
            camera = cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)

            // プレビュー用の SurfaceProviderをビューファインダーに設定
            preview?.setSurfaceProvider(fragmentMainBinding.viewFinder.surfaceProvider)
            hassurfaceprovider = true
        }
        catch (exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
            hassurfaceprovider = false
        }
    }


    // CameraXのプレビュー画像を基に、ObjectdetectHelperにて推論を実行
    // @param image CameraXの解析用イメージ 640*480*3
    // @param pitch デバイスの傾き
    private fun detectObjects(image: ImageProxy, pitch: Int) {

        image.use { bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer) }

        // 画像の回転情報 (角度) の取得
        val imageRotation = image.imageInfo.rotationDegrees

        objectDetector.detect(bitmapBuffer, imageRotation, pitch)
    }


    // デバイスの画面の向きが変わった際に呼び出し
    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        imageAnalyzer?.targetRotation = fragmentMainBinding.viewFinder.display.rotation
    }


    // ObjectDetectorから結果を受け取る
    // CameraExecutorThread 上での処理であることに注意
    // 推論結果をPythonモジュールに渡し、マスクを作成し、イメージビューを作成する
    // @param out1 1*38*8400　推論スコア
    // @param out2 1*32*160*160　推論マスク
    // @param pitch デバイスの傾き
    // @param inputstyle 1*3*640*640 正規化済み解析用イメージ
    override fun onResults(
        out1: Array<Array<FloatArray>>,
        out2: Array<Array<Array<FloatArray>>>,
        pitch: Int,
        inputstyle: Array<Array<Array<FloatArray>>>
    ) {

        // Executorで実行　
        // キューに処理が溜まらないよにBooleanでブロック
        if (pyIsRunnable) {
            pyIsRunnable = false
            calpisExecutor.execute {

                // chaquopyでは元の配列を直接参照する点に注意
                // なので、.arrayでコピーを作成し、pyモジュールに渡す

                // out1
                val byteBuffer1 = ByteBuffer.allocate(38 * 8400 *4)
                byteBuffer1.order(ByteOrder.nativeOrder())
                val floatBufView1 = byteBuffer1.asFloatBuffer()
                for (i in 0 until 38){
                    for (j in 0 until 8400){
                        floatBufView1.put(out1[0][i][j])
                    }
                }

                // out2
                val byteBuffer2 = ByteBuffer.allocate(32 * 160 * 160 *4)
                byteBuffer2.order(ByteOrder.nativeOrder())
                val floatBufView2 = byteBuffer2.asFloatBuffer()
                for (i in 0 until 32){
                    for (j in 0 until 160){
                        for (k in 0 until 160) {
                            floatBufView2.put(out2[0][i][j][k])
                        }
                    }
                }

                // 1*3*640*640　-> 640*640*3 に戻す
                val byteBuffer3 = ByteBuffer.allocate(3 * 640 * 640 *4)
                byteBuffer3.order(ByteOrder.nativeOrder())
                val floatBufView3 = byteBuffer3.asFloatBuffer()
                for (i in 0 until 3){
                    for (j in 0 until 640){
                        for (k in 0 until 640){
                            floatBufView3.put(inputstyle[0][i][j][k])
                        }
                    }
                }

                // Pythonインスタンスの取得
                val py = Python.getInstance()
                val pymodule = py.getModule("pymodule")

                /*
                    keyでPythonモジュールの関数名を指定
                    @return  [ int , str or bytearray(PNG)]
                    int == 1 -> 正常に検出 / bytearray
                    int == 2 -> オーバーフロー / str ログ
                    int == 3 -> 未検出 / str ログ

                    main/python/pymodule.py を呼び出し
                    https://chaquo.com/chaquopy/
                */
                val cv2result = pymodule.callAttr(
                    "entry_point",
                    execMode,
                    byteBuffer3.array(),
                    byteBuffer1.array(),
                    byteBuffer2.array(),
                    layoutHeight,
                    layoutWidth,
                    pitch,
                    concentration,
                    glassVolume,
                    thickBottom
                ).asList()


                // CameraXプレビューが表示されている & 画面が停止されていない & cv2result == 1　の場合
                // -> 正常に検出されたので、calpismask / contentsxy座標 / holdcount を更新
                if (hassurfaceprovider &&nonfreeze&& cv2result[0].toJava(Int::class.java) == 1) {

                    val imgBytearray = cv2result[1].toJava(ByteArray::class.java)

                    // ARGB_8888に注意
                    val op = BitmapFactory.Options()
                    op.inPreferredConfig = Bitmap.Config.ARGB_8888
                    op.inMutable = true
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                        op.inPreferredColorSpace = ColorSpace.get(ColorSpace.Named.SRGB);
                    }

                    // ByteArrayをbitmapにデコード
                    calpismask = BitmapFactory.decodeByteArray(imgBytearray, 0, imgBytearray.size,op)

                    // 検出された内容物の画面上XY座標
                    contentsx = cv2result[2].toJava(Int::class.java)
                    contentsy = cv2result[3].toJava(Int::class.java)

                    // 内容物イメージの画面表示カウンターをリセット
                    holdcount = 0

                }

                // CameraXプレビューが表示されていない | 画面が停止されている 場合
                // -> 更新はなし
                else if (!hassurfaceprovider || !nonfreeze) {
                }

                // cv2result == 2　の場合
                // -> グラスから溢れている為、エラー表示
                else if (cv2result[0].toJava(Int::class.java) == 2) {

                    // オーバーフローアニメーションの更新
                    if (overflowanime.isRunning) {
                        overflowanime.stop()
                    }
                    overflowanime.start()

                    Log.e(TAG, cv2result[1].toJava(String::class.java))

                }

                // cv2result == 3　の場合
                // -> オブジェクト未検出
                else if(cv2result[0].toJava(Int::class.java) == 3){
                    Log.e(TAG, cv2result[1].toJava(String::class.java))
                }

                // エラー
                else{
                    Log.e(TAG, cv2result[1].toJava(String::class.java))

                    // エラーアニメーションの更新
                    if(erroranime.isRunning){
                        erroranime.stop()
                    }
                    erroranime.start()
                }

                // 処理が完了したので、再進入可能
                pyIsRunnable = true
            }
        }

        // 一定時間経過後にcalpismask を null
        if (hassurfaceprovider && nonfreeze && holdcount > 3){ calpismask = null }
        holdcount++
    }


    override fun onError(error: String) {
        activity?.runOnUiThread {
            Toast.makeText(requireContext(), error, Toast.LENGTH_SHORT).show()
        }
    }


    override fun onPause() {
        super.onPause()
        // センサリスナーの破棄
        if (mIsMagSensor || mIsAccSensor) {
            mSensorManager.unregisterListener(this)
            mIsMagSensor = false
            mIsAccSensor = false
        }
    }


    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}


    // 各種センサーの値の変化を検知
    override fun onSensorChanged(event: SensorEvent) {

        when (event.sensor.type) {
            // 地磁気センサー
            Sensor.TYPE_MAGNETIC_FIELD ->
                mMagneticValues = event.values.clone()
            // 加速度センサー
            Sensor.TYPE_ACCELEROMETER ->
                mAccelerometerValues = event.values.clone()
            else ->
                return
        }

        // 回転行列と傾斜行列の算出
        if (mMagneticValues != null && mAccelerometerValues != null) {
            val rotationMatrix = FloatArray(matrixSIZE)
            val inclinationMatrix = FloatArray(matrixSIZE)
            val remapedMatrix = FloatArray(matrixSIZE)
            val orientationValues = FloatArray(dimension)

            SensorManager.getRotationMatrix(
                rotationMatrix,
                inclinationMatrix,
                mAccelerometerValues,
                mMagneticValues
            )
            SensorManager.remapCoordinateSystem(
                rotationMatrix,
                SensorManager.AXIS_X,
                SensorManager.AXIS_Z,
                remapedMatrix
            )
            SensorManager.getOrientation(remapedMatrix, orientationValues)

            //mAzimuthZ: 方位角 (Z軸回転)
            //mPitchX: ピッチ (X軸回転)
            //mRollY: ロール (Y軸回転)
            mAzimuthZ = (orientationValues[0]*rad2deg).toInt()
            mPitchX = (orientationValues[1]*rad2deg).toInt()
            mRollY = (orientationValues[2]*rad2deg).toInt()
        }
    }


    // FloatingIceの挙動を設定
    private fun setYoYo(){
        YoYo.with(Techniques.Swing)
            .duration(19000)
            .repeat(YoYo.INFINITE)
            .playOn(activity?.findViewById<View>(R.id.icecube1))
        YoYo.with(Techniques.Swing)
            .duration(23000)
            .repeat(YoYo.INFINITE)
            .playOn(activity?.findViewById<View>(R.id.icecube2))
        YoYo.with(Techniques.Swing)
            .duration(25000)
            .repeat(YoYo.INFINITE)
            .playOn(activity?.findViewById<View>(R.id.icecube3))
        YoYo.with(Techniques.Swing)
            .duration(20000)
            .repeat(YoYo.INFINITE)
            .playOn(activity?.findViewById<View>(R.id.icecube4))
        YoYo.with(Techniques.Swing)
            .duration(19000)
            .repeat(YoYo.INFINITE)
            .playOn(activity?.findViewById<View>(R.id.icecube5))
        YoYo.with(Techniques.Swing)
            .duration(21000)
            .repeat(YoYo.INFINITE)
            .playOn(activity?.findViewById<View>(R.id.icecube6))
        YoYo.with(Techniques.Swing)
            .duration(20000)
            .repeat(YoYo.INFINITE)
            .playOn(activity?.findViewById<View>(R.id.icecube7))
        YoYo.with(Techniques.Swing)
            .duration(23000)
            .repeat(YoYo.INFINITE)
            .playOn(activity?.findViewById<View>(R.id.icecube8))
        YoYo.with(Techniques.Swing)
            .duration(22000)
            .repeat(YoYo.INFINITE)
            .playOn(activity?.findViewById<View>(R.id.icecube9))
        YoYo.with(Techniques.Shake)
            .duration(900)
            .repeat(YoYo.INFINITE)
            .playOn(activity?.findViewById<View>(R.id.overflow_icon))
        YoYo.with(Techniques.Shake)
            .duration(700)
            .repeat(YoYo.INFINITE)
            .playOn(activity?.findViewById<View>(R.id.error_icon))
    }
}
