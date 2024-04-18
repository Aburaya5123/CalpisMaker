package jp.gr.java_conf.calpismaker

import android.os.Build
import android.os.Bundle
import android.view.View
import android.widget.LinearLayout
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import jp.gr.java_conf.calpismaker.databinding.ActivityMainBinding


/*
    Fragmentトランザクションは以下参照

    res/navigation/nav_graph.xml
 */
class MainActivity : AppCompatActivity() {

    private lateinit var activityMainBinding: ActivityMainBinding


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // アプリのコンテンツをシステムバーにかぶさるように設定
        WindowCompat.setDecorFitsSystemWindows(window, false)

        // ステータスバーを非表示に設定
        val windowInsetsCompat = WindowInsetsControllerCompat(window, window.decorView)
        windowInsetsCompat.hide(WindowInsetsCompat.Type.statusBars())
        windowInsetsCompat.systemBarsBehavior =
            WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE

        // スワイプによってステータスバーの表示が切り替わるように設定
        val rootid = window.decorView.findViewById<View>(android.R.id.content)
        ViewCompat.setOnApplyWindowInsetsListener(
            rootid
        ) { view: View, windowInsets: WindowInsetsCompat ->
            val insets = windowInsets.getInsets(WindowInsetsCompat.Type.systemBars())
            view.layoutParams = (view.layoutParams as LinearLayout.LayoutParams).apply {
                bottomMargin = insets.bottom
            }
            WindowInsetsCompat.CONSUMED
        }

        // Appのテーマを指定　->　res/values/styles.xml
        setTheme(R.style.AppTheme)

        Thread.sleep(2000)

        activityMainBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(activityMainBinding.root)
    }

    @Deprecated("Deprecated in Java")
    override fun onBackPressed() {

        // apiレベルによって処理を切り替え
        if (Build.VERSION.SDK_INT == Build.VERSION_CODES.Q) {
            finishAfterTransition()
        } else {
            super.onBackPressed()
        }
    }
}
