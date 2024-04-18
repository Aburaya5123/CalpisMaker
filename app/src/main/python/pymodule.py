import numpy as np
import cv2
import torch
import torchvision
import torch.nn.functional as F
import math


def entry_point(exec_mode,rawarray,input1,input2,layout_h,layout_w,pitch,concentration
                ,glass_volume,thick_bottom):
    """
    Androidから受け取ったyoloの推論結果をもとに、画面に表示する内容物のイメージを作成する

    java <-> Python 間で受け渡しのできるオブジェクトは以下参照
    https://chaquo.com/chaquopy/doc/15.0/python.html

    :param exec_mode: int
        動作モードを表す　1-> 希釈モード　, 2-> 原液モード
    :param rawarray: bytearray[model_size][model_size][3]
        CameraXのグレーパディング済み解析画像 640*640*3
    :param input1: bytearray[batch_size][num_classes + 4 + num_masks, num_boxes][ box , class , mask1, mask2 ...]
        推論スコア 1*38*8400
    :param input2: bytearray[batch_size][num_classes][mask_size][mask_size]
        推論 mask 1*32*160*160
    :param layout_h: int
        デバイスディスプレイの高さ
    :param layout_w: int
        デバイスディスプレイの横幅
    :param pitch: int
        デバイスの傾き
    :param concentration: int
        濃度設定値
    :param glass_volume: int
        容量設定値
    :param thick_bottom: int
        0 -> 厚底でない , 1 -> 厚底

    :return: List[int, bytes, int, int] / List[int, str]
        希釈モード時 :
            グラスが見つかり、内容物も確認 -> [int(1), bytes(内容物のイメージ), int(x座標), int(y座標)]
            グラスが見つかったが、内容物が見つからなかった -> [int(3), str(ログ)]
            グラスが見つかったが、内容物がグラスから溢れた -> [int(2), str(ログ)]
        原液モード時 :
            グラスが見つかった -> [int(1), bytes(内容物のイメージ), int(x座標), int(y座標)]
            グラスが見つからなかった -> [int(3), str(ログ)]
        エラー -> [int(0), str(ログ)]
    """

    # ディスプレイサイズが取得できていなければ返す
    if layout_h==0 or layout_w==0:
        return [0,"PyModule : Display size not acquired.[0]"]

    # 出力比率
    scale_ratio = max(layout_h/640, layout_w/480)

    # rawarray の中心座標を求める
    wh_ratio = layout_w / layout_h
    screen_w = 640 * wh_ratio
    screen_c = [320,screen_w//2]

    # rawarray を ndarray に変換
    np_img = np.frombuffer(rawarray,np.float32).reshape(3,640,640)

    # yolo1 -> 推論スコア
    # yolo2 -> 推論 mask
    yolo1 = np.frombuffer(input1,np.float32).reshape(1,38,8400)
    yolo2 = np.frombuffer(input2,np.float32).reshape(1,32,160,160)

    # torch.Tensorに変換
    yolo_ts1 = torch.from_numpy(yolo1)

    # NMSの実行
    # ↑ 複数の検出結果の中から最も高いスコアを持つ結果を残す
    nms = non_max_suppression(yolo_ts1)

    if len(nms[0]) == 0:
        return [0,"PyModule : No object was detected.[0]"]

    proto = yolo2[0]

    # 検出したグラスの中心座標と、bbox を格納する dictionary
    glass_coords = {} # glass_coorde[glass_enum] = [[x_center, y_center], x1, y1, x2, y2]

    # 検出した内容物の中心座標と、bboxを格納する dictionary
    mask_coords = {} # mask_coorde[mask_enum] = [[x_center, y_center)], x1, y1, x2, y2]

    # グラスの bbox 内に、内容物の中心座標がある場合、該当する内容物の mask_enum と 中心座標を格納する dictionary
    inside_glass = {} # inside_glass[glass_enum] = [[mask_enum,[mask_center_x,mask_center_y]],...]

    mask = process_mask(torch.from_numpy(proto), nms[0][:,6:], nms[0][:,:4], (640,640), upsample=True)

    # グラスの bbox を 640*480 にスケーリング後、glass_coord に glass_enum をキーに格納
    for i, pred in enumerate(nms[0]):

        # グラスの class == 1 であるので、pred[5] == 1 であればグラスを検出
        if int(pred[5])==1:

            pred[:4] = scale_boxes((640,640), pred[:4], (640,480)).round()
            # ObjectDetector で左右にパディングした分 bbox をずらす (80pixel）
            gc = [int(pred[0].item())+80,int(pred[1].item()),int(pred[2].item())+80,int(pred[3].item())] # bbox
            gc.insert(0,[gc[0]+(gc[2]-gc[0])//2,gc[1]+(gc[3]-gc[1])//2]) # gc = [[x_center, y_center], x1, y1, x2, y2]
            glass_coords[i] = gc

    # グラスを検出できなかった場合
    if not glass_coords:
        return [3,"PyModule : Glass not found.[1]"]


    # 内容物の中心座標がグラスの bbox の範囲内にあれば容器内にあると判断し、inside_glass に追加する
    for i, pred in enumerate(nms[0]):

        # 内容物の class == 0 であるので、pred[5] == 0 であれば内容物を検出
        if int(pred[5])==0:

            pred[:4] = scale_boxes((640,640), pred[:4], (640,480)).round()
            # ObjectDetector で左右にパディングした分 bbox をずらす (80pixel）
            mc = [int(pred[0].item())+80,int(pred[1].item()),int(pred[2].item())+80,int(pred[3].item())] # bbox
            mask_center = [mc[0]+(mc[2]-mc[0])//2,mc[1]+(mc[3]-mc[1])//2]
            mask_coords[i] = mc
            for j in glass_coords.keys():
                gcv = glass_coords[j]
                # mask の中心点がグラスの bbox 内にあるので、inside_glass に追加
                if gcv[1]<mask_center[0]<gcv[3] and gcv[2]<mask_center[1]<gcv[4]:
                    inside_glass.setdefault(j, []).append([i,mask_center])


    # 原液モード : 画面の中心に最も近い位置にあるグラスを選択
    if exec_mode==2:
        midmost = get_midmost_coord(glass_coords, glass_coords, screen_c)

    # 希釈モード : 内容物が見つからない場合
    elif not inside_glass:
        return [3,"PyModule : Calpis not found.[4]"]

    # 希釈モード : inside_glass の enum の中から、画面の中心に最も近い位置にある内容物を選択
    else:
        midmost = get_midmost_coord(inside_glass, glass_coords, screen_c)

    if midmost[1]==999999:
        return [0,"PyModule : The glass is out of the screen bounds.[5]"]


    glass_mask = mask[midmost[0]].numpy().astype(np.uint8)*255 # 0~1 -> 0~255

    # 原液モードではこの処理をスキップ
    if exec_mode==1:
        bottom_contents = [0,-1] # bottom_contents[mask_enum, height]
        # 内容物が複数検出された場合、中心座標y が最も大きい mask を選択
        for i in inside_glass[midmost[0]]:
            if i[1][1] > bottom_contents[1]:
                bottom_contents[0] = i[0]
                bottom_contents[1] = i[1][1]

        if bottom_contents[1]==-1:
            return [0,"PyModule : The content is out of the glass bounds.[6]"]

        contents_mask = mask[bottom_contents[0]].numpy().astype(np.uint8)*255 # 0~1 -> 0~255


    _, glass_mask = cv2.threshold(glass_mask, 160, 255, cv2.THRESH_BINARY)

    # 原液モードと処理分岐
    if exec_mode==2:
        return measure_volume(glass_coords[midmost[0]],scale_ratio,glass_mask,pitch,concentration,glass_volume,thick_bottom)

    _, contents_mask = cv2.threshold(contents_mask, 160, 255, cv2.THRESH_BINARY)
    glass_mask[contents_mask==255] = 255


    # 結果として表示する内容物の色を作成
    mode_1, mode_2 = get_content_color(np_img, contents_mask)
    np_img = None


    # ディスプレイ中央に最も近い位置にあるグラスと内容物の bbox
    glass_box = glass_coords[midmost[0]][1:5] # xyxy
    contents_box = mask_coords[bottom_contents[0]] # xyxy

    if any(i<0 | 640<i for i in glass_box) or any(i<0 | 640<i for i in contents_box):
        return [0,"PyModule : The bbox size is incorrect.[7]"]


    # グラス飲み口の楕円を計算
    glass_top_deg = calc_ratio(glass_box[1],pitch)
    if glass_top_deg == 9999:
        return [0,"PyModule : Top of the glass is out of the bound.[8]"]
    elif glass_top_deg!=0:
        glass_rim_r = int((glass_box[2] - glass_box[0]) * glass_top_deg /2)
        glass_rim_w = cv2.countNonZero(glass_mask[glass_box[1]+glass_rim_r]) # グラス飲み口の楕円長径
        glass_rim = int(glass_rim_w * glass_top_deg /2) # グラス飲み口の楕円短径/2
    else:
        glass_rim = 0


    # 内容物表面の楕円を計算
    contents_surface = contents_box[1]
    contents_deg = calc_ratio(contents_surface,pitch)
    if contents_deg == 9999:
        return [0,"PyModule : Top of the content is out of the bound.[9]"]
    elif contents_deg!=0:
        contents_sr = int((contents_box[2]-contents_box[0]) * contents_deg) # 表面楕円短径
        if contents_sr+contents_surface>contents_box[3]:
            return [0,"PyModule : The content size is incorrect.[10]"]
        contents_surface += contents_sr//2 # 表面楕円中心点y座標
        contents_height = contents_box[3] - contents_box[1] - contents_sr # 内容物の高さ
    else:
        contents_sr = 0
        contents_height = contents_box[3] - contents_box[1]

    if contents_height < 0:
        return [0, "PyModule : The content size is incorrect.[11]"]

    # 内容物の中心x座標を求める
    c_width = np.where(glass_mask[contents_surface]==255)[0]
    if c_width!=[]:
        contents_gx = c_width[len(c_width)//2]
    else:
        contents_gx = (contents_box[0]+contents_box[2])//2

    # 内容物の mask から、表面の楕円形を切り抜く
    cv2.ellipse(contents_mask, ((contents_gx, contents_surface),
                                (contents_box[2]-contents_box[0], contents_sr), 0), (0, 0, 0), thickness=-1)
    contents_mask[:contents_surface] = 0


    # 内容物の側面の輪郭取得
    contents_contours2 = cv2.findContours(contents_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    if contents_contours2:
        contents_contours2 = max(contents_contours2, key=lambda x: cv2.contourArea(x))


    # 倍率が1の場合は元の内容物のイメージを出力
    if concentration==1:
        result = np.zeros((640,640,4)) # 出力イメージ
        # 内容物の色で mask を塗りつぶす
        result[contents_mask==255] = (mode_2[2],mode_2[1],mode_2[0],200)
        # 輪郭を描写
        cv2.drawContours(result, contents_contours2, -1, (0,5,58,120), 1,cv2.LINE_AA)
        # グラスの範囲にイメージを切り抜き
        glassx = (glass_box[0]-80)*scale_ratio
        glassy = glass_box[1]*scale_ratio
        result = result[glass_box[1]:glass_box[3],glass_box[0]:glass_box[2]]
        # ディスプレイのサイズにイメージを拡大
        result = cv2.resize(result,dsize=None,fx=scale_ratio,fy=scale_ratio)
        # png型式でエンコードし、bytearray で返す
        ret, encoded = cv2.imencode(".png", result)
        return [1,encoded.tobytes(),int(glassx),int(glassy)]


    # 内容物の底からグラス飲み口へ向かって、pixel数をカウントしていきながら、仮体積を求める
    startline = int(contents_box[3] - contents_sr/2)
    endline = glass_box[1]+glass_rim
    gap = int(contents_sr/2)

    undiluted = 0
    for i in range(startline,startline-contents_height,-1):
        gw = cv2.countNonZero(glass_mask[i])
        undiluted += (gw/2)**2 * math.pi

    # カウントした仮体積に、計算した倍率を掛けて、表示するイメージの仮体積を求める
    undiluted *= (concentration-1) / math.sqrt(1-(contents_deg**2)) * 1.15

    diluted = 0
    for j in range(i,endline,-1):
        gw = cv2.countNonZero(glass_mask[j])
        diluted += (gw/2)**2 * math.pi
        if diluted >= undiluted:
            target_h = j
            break
    else:
        return [2,"PyModule : Overflow.[12]"]

    target_h += gap # startline から　target_h までが内容物の側面の高さとなる


    # グラス底の楕円を計算
    bottom_deg = calc_ratio(contents_box[3],pitch)
    if bottom_deg==9999:
        return [0,"PyModule : Bottom of the content is out of the bound.[13]"]
    elif bottom_deg!=0:
        bottom_w = cv2.countNonZero(glass_mask[(contents_box[1] + contents_sr +contents_box[3])//2])
        bottom_r = int(bottom_w * bottom_deg /2)
        bottom_w = cv2.countNonZero(glass_mask[contents_box[3]-bottom_r]) # 底楕円の長径
        bottom_r = int(bottom_w * bottom_deg /2) # 底楕円の短径/2
    else:
        bottom_r = 0
        bottom_w = cv2.countNonZero(glass_mask[contents_box[3]])


    # 出力イメージの表面楕円を計算
    target_deg = calc_ratio(target_h,pitch)
    if target_deg==9999:
        return [0,"PyModule : Top of the content is out of the bound.[14]"]
    elif target_deg!=0:
        target_w = cv2.countNonZero(glass_mask[target_h])
        target_r = int(target_w * target_deg/2) # 表面楕円短径/2
        target_h -= target_r # 表面楕円中心点y座標
        target_w = cv2.countNonZero(glass_mask[target_h]) # 楕円長径
    else:
        target_w = cv2.countNonZero(glass_mask[target_h])
        target_r = 0

    t_width = np.where(glass_mask[target_h]==255)[0] # 楕円の中心点のy方向に、255 のpixel数をカウント
    if t_width!=[]:
        rim_gx = t_width[len(t_width)//2] # 楕円の中心x座標
    else:
        rim_gx = (contents_box[0]+contents_box[2])//2


    # 出力イメージの底面楕円の中心ｙ座標
    b_width = np.where(glass_mask[contents_box[3]-bottom_r]==255)[0]
    if b_width!=[]:
        bottom_gx = b_width[len(b_width)//2]
    else:
        bottom_gx = (contents_box[0]+contents_box[2])//2


    # 出力イメージの底面楕円を描写
    temp_mask = np.zeros(glass_mask.shape)
    cv2.ellipse(temp_mask, ((bottom_gx, contents_box[3]-bottom_r),
                             (bottom_w, bottom_r*2), 0), (255, 255, 255), thickness=-1)
    glass_mask[contents_box[3]-bottom_r:] = temp_mask[contents_box[3]-bottom_r:]

    glass_mask2 = np.zeros((640,640))
    glass_mask2[glass_mask==255] = 255 # copy

    # glass_mask2 に出力イメージの表面楕円を描写し、その輪郭を取得する
    glass_mask2 = cv2.ellipse(glass_mask2, ((contents_gx, contents_surface),
                             (cv2.countNonZero(glass_mask[contents_surface])
                              , contents_sr), 0), 0, thickness=-1)
    glass_mask2[:contents_surface]=0
    contents_contours = cv2.findContours(glass_mask2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    if contents_contours:
        contents_contours = max(contents_contours, key=lambda x: cv2.contourArea(x))

    # glass_mask に出力イメージの表面楕円を除いた図形を描写し、その輪郭を取得する
    cv2.ellipse(glass_mask, ((rim_gx, target_h),
                              (target_w,target_r*2), 0), 0, thickness=-1)
    glass_mask[:target_h] = 0
    glass_contours = cv2.findContours(glass_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    if glass_contours:
        glass_contours = max(glass_contours, key=lambda x: cv2.contourArea(x))


    result = np.zeros((640,640,4))

    result[glass_mask==255] = (mode_1[2],mode_1[1],mode_1[0],140)
    result[glass_mask2==255] = (mode_2[2],mode_2[1],mode_2[0],200)

    # 検出した内容物の輪郭線と、出力イメージの輪郭線を描写
    cv2.drawContours(result,contents_contours, -1, (0,5,58,120), 1,cv2.LINE_AA)
    cv2.drawContours(result, glass_contours, -1, (0,5,58,120), 1,cv2.LINE_AA)

    return output_process(glass_box, result, scale_ratio)



def get_content_color(color_img, mask):

    """
    CameraXの画像から、内容物の mask をとる
    そこから内容物の色を取得し、彩度を調整したのち、RGBで値を返す

    :param color_img: ndarray[3, 640, 640]
        CameraXの画像 RGB
    :param mask: ndarray[640, 640]
        内容物に該当する範囲は 255, それ以外は 0

    :return: numpy array[R, G, B], numpy array[R, G, B]
        mode_1 -> 元の色より彩度が高い
        mode_2 -> 元の色より彩度が低い
    """

    # カラーの画像から内容物を指すピクセルを選択し、RGBの平均をとる
    rmean = color_img[0][mask==255] # 0~1
    gmean = color_img[1][mask==255]
    bmean = color_img[2][mask==255]

    pix_num = rmean.shape[0]
    rmean = sum(rmean)/pix_num
    gmean = sum(gmean)/pix_num
    bmean = sum(bmean)/pix_num

    mode_color = np.array([[[int(bmean*255),int(gmean*255),int(rmean*255)]]], dtype=np.uint8) #0~255

    # RGB から HLS へ変換し、彩度を上げた色と彩度を下げた色の2色を作成
    mode_hls = cv2.cvtColor(mode_color, cv2.COLOR_BGR2HLS)
    mode_hls2 = mode_hls.copy()
    if mode_hls[0,0,1] + 20 <255:
        mode_hls[0,0,1] += 20
    else:
        mode_hls[0,0,1] =255
    if mode_hls2[0,0,1] -20 > 0:
        mode_hls2[0,0,1] -= 20
    else:
        mode_hls2[0,0,1] = 0
    mode_1 = cv2.cvtColor(mode_hls, cv2.COLOR_HLS2RGB).flatten()
    mode_2 = cv2.cvtColor(mode_hls2, cv2.COLOR_HLS2RGB).flatten()

    return mode_1, mode_2



def get_midmost_coord(coords1, coords2, screen_c):

    """
    coords 内の座標から、screen_c 座標に最も近いものを探す

    :param coords1: dictionary{[[int, int)], float, float, float, float], ...}
        中心座標と bbox の List が格納された辞書　[[x_center, y_center)], x1, y1, x2, y2]
    :param coords2: dictionary{[[int, int)], float, float, float, float], ...}
        中心座標と bbox の List が格納された辞書　[[x_center, y_center)], x1, y1, x2, y2]
    :param screen_c: List[int, int]
        ディスプレイの中心座標

    :return: List[int, float]
        midmost[glass_enum, distance]
    """

    midmost = [0,999999]
    for i in coords1.keys():
        g_center = coords2[i][0]
        distance = (g_center[0] - screen_c[0])**2 + (g_center[1] - screen_c[1])**2
        if distance < midmost[1]:
            midmost[0] = i
            midmost[1] =  distance

    return midmost



def measure_volume(glass_coords,scale_ratio,mask,pitch,concentration,glass_volume,thick_bottom):

    """
    原液モード
    希釈モードと基本処理は同じ

    :param glass_coords: List[[int, int], int, int, int, int]
        グラスの中心座標と bbox [[x_center, y_center], x1, y1, x2, y2]
    :param scale_ratio: float
        出力比率
    :param mask: ndarray[640, 640]
        グラスの mask
    :param pitch: int
        デバイスの傾き
    :param concentration: int
        濃度設定値
    :param glass_volume: int
        容量設定値
    :param thick_bottom: int
        0 -> 厚底でない , 1 -> 厚底

    :return: List[int, bytes, int, int] / List[int, str]
        希釈モード時 :
            グラスが見つかり、内容物も確認 -> [int(1), bytes(内容物のイメージ), int(x座標), int(y座標)]
            グラスが見つかったが、内容物が見つからなかった -> [int(3), str(ログ)]
            グラスが見つかったが、内容物がグラスから溢れた -> [int(2), str(ログ)]
        原液モード時 :
            グラスが見つかった -> [int(1), bytes(内容物のイメージ), int(x座標), int(y座標)]
            グラスが見つからなかった -> [int(3), str(ログ)]
        エラー -> [int(0), str(ログ)]
    """

    glass_box = glass_coords[1:5]

    # グラスの中心x座標を求める
    c_idx = np.where(mask[(glass_box[1]+glass_box[3])//2]==255)[0]
    if c_idx!=[]:
        center_line = c_idx[len(c_idx)//2]
    else:
        center_line = (glass_box[0]+glass_box[2])//2


    rim_w =  glass_box[2] - glass_box[0] # グラス飲み口の楕円長径
    rim_deg = calc_ratio(glass_box[1],pitch)
    if rim_deg==9999:
        return [0, "PyModule : Top of the glass is out of the bound.[14]"]
    elif rim_deg!=0:
        rim_r = int(rim_w * rim_deg/2)
        rim_w = cv2.countNonZero(mask[glass_box[1] + rim_r])
        rim_r = int(rim_w * rim_deg) # グラス飲み口の楕円短径
        rim_h = glass_box[1] + rim_r # グラス飲み口の楕円中心y座標
        h1 = glass_box[3] - rim_h
    else:
        rim_r = 0
        h1 = glass_box[3] - glass_box[1]
        rim_h = glass_box[1]


    # 厚底なら底の高さを加算
    if thick_bottom==0:
        bottom_line = int(rim_h + (h1 * 0.95))
    else:
        bottom_line = int(rim_h + (h1 * 0.8))


    # グラス底の楕円計算
    c_deg = calc_ratio(((glass_box[3] + glass_box[1])//2+glass_box[3])//2, pitch)
    c_width = cv2.countNonZero(mask[((glass_box[3] + glass_box[1])//2+glass_box[3])//2])
    if c_deg==9999:
        return [0,"PyModule : Bottom of the glass is out of the bound.[15]"]
    elif c_deg!=0:
        c_r = int(c_width * c_deg /2) # グラス底の楕円短径/2
    else:
        c_r = 0

    b_deg = calc_ratio(bottom_line- c_r, pitch)
    b_width = cv2.countNonZero(mask[bottom_line-c_r])
    if b_deg==9999:
        return [0,"PyModule : Bottom of the glass is out of the bound.[16]"]
    elif b_deg!=0:
        b_r = int(b_width * b_deg /2)
        b_width = cv2.countNonZero(mask[bottom_line-b_r])
        b_r = int(b_width * b_deg /2) # グラス底の楕円短径/2
        b_width = cv2.countNonZero(mask[bottom_line-b_r]) # グラス底の楕円長径
    else:
        b_r = 0
        b_width = cv2.countNonZero(mask[bottom_line])


    # グラス底からグラス飲み口へ向かって、pixel数をカウントしていきながら、仮体積を求める
    start_height = bottom_line - b_r
    end_heght = (glass_box[1] + rim_h)//2
    height_gap = b_r

    full_cap = 0
    for i in range(start_height,end_heght,-1):
        w_width = cv2.countNonZero(mask[i])
        full_cap += (w_width/2)**2 * math.pi

    # カウントした値に、計算した倍率を掛けて、表示するイメージの仮体積を求める
    target_vol = full_cap * glass_volume / concentration * 0.1

    volume = 0
    for i in range(start_height,end_heght,-1):
        w_width = cv2.countNonZero(mask[i])
        volume += (w_width/2)**2 * math.pi
        if volume>=target_vol:
            waterline = i
            break
    else:
        return [2,"PyModule : Overflow.[17]"]

    waterline += height_gap
    if waterline>bottom_line:
        return [0,"PyModule : The water line is incorrect.[18]"]


    # 出力イメージの表面の楕円を計算
    surface_degrees = calc_ratio(waterline,pitch)
    if surface_degrees == 9999:
        return [0,"PyModule : Top of the glass is out of the bound.[19]"]
    elif surface_degrees!=0:
        perspective = math.sqrt(1-(surface_degrees**2))
        if perspective!=0:
            waterline = waterline + int((bottom_line - waterline)*(1-perspective))
        elif perspective < 0:
            return [0,"PyModule : The perspective is incorrect.[20]"]

        surface_w = cv2.countNonZero(mask[waterline])
        surface_r = int(surface_w * surface_degrees/2) # 表面楕円の短径/2
        waterline -= surface_r #　出力イメージy軸上で最も大きい値
        surface_w = cv2.countNonZero(mask[waterline]) # 表面楕円の長径
    else:
        surface_r = 0
        surface_w = cv2.countNonZero(mask[waterline])


    # グラス底の中心x座標
    bc_idx = np.where(mask[bottom_line-b_r]==255)[0]
    if bc_idx!=[]:
        b_center_line = bc_idx[len(bc_idx)//2]
    else:
        b_center_line = center_line

    # グラス飲み口の中心x座標
    sc_idx = np.where(mask[waterline]==255)[0]
    if sc_idx!=[]:
        s_center_line = sc_idx[len(sc_idx)//2]
    else:
        s_center_line = center_line

    # グラス底の楕円を描写
    bottom_mask = np.zeros(mask.shape)
    cv2.ellipse(bottom_mask, ((b_center_line, bottom_line-b_r), (b_width, b_r*2), 0), (255,255,255), thickness=-1)
    mask[bottom_line-b_r:] = bottom_mask[bottom_line-b_r:]
    cv2.ellipse(mask, ((s_center_line, waterline), (surface_w, surface_r*2), 0), (0,0,0), thickness=-1)

    mask[:waterline]=0

    result = np.zeros((640,640,4))
    result[mask==255]= (255,255,255,180)

    # 出力イメージの輪郭を描写
    contents_contours2 = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    if contents_contours2:
        contents_contours2 = max(contents_contours2, key=lambda x: cv2.contourArea(x))
    cv2.drawContours(result, contents_contours2, -1, (0,5,58,120), 1,cv2.LINE_AA)

    return output_process(glass_box, result, scale_ratio)



def output_process(glass_box, result, scale_ratio):

    """
    出力前のサイズ調整

    :param glass_box: List
        グラスの bbox xyxy
    :param result: ndarray[640,640,4]
        出力イメージ
    :param scale_ratio: float
        出力比率

    :return: List[int, bytes, int, int] / List[int, str]
        希釈モード時 :
            [int(1), bytes(内容物のイメージ), int(x座標), int(y座標)]
        原液モード時 :
            [int(1), bytes(内容物のイメージ), int(x座標), int(y座標)]
    """

    # 画面端とのマージン確保
    if glass_box[1]-5>=0:
        glass_box[1] -= 5
    if glass_box[3]+5<640:
        glass_box[3] += 5

    result = result[glass_box[1]:glass_box[3],glass_box[0]-5:glass_box[2]+5]

    result = cv2.resize(result,dsize=None,fx=scale_ratio,fy=scale_ratio)
    glassx = (glass_box[0]-80-5)*scale_ratio # 出力イメージのディスプレイ上x座標
    glassy = glass_box[1]*scale_ratio # 出力イメージのディスプレイ上y座標
    ret, encoded = cv2.imencode(".png", result)

    return [1,encoded.tobytes(),int(glassx),int(glassy)]



def non_max_suppression(
        prediction,
        conf_thres = 0.6, # 0.25
        iou_thres = 0.5,
        agnostic = False,
        max_det = 10,
        nm = 32,
):
    """
    NMSで重複する検出結果を省く
    参考 :
    https://github.com/ultralytics/ultralytics/blob/d608565a17384fe8bd666bd91318191846b50cbc/ultralytics/utils/ops.py#L162

    :param prediction: torch.Tensor[batch_size][num_classes + 4 + num_masks, num_boxes][...]
        推論結果のスコア (1*38*8400)
            [batch_size][num_classes + 4 + num_masks, num_boxes][ box , class , mask1, mask2 ...]
    :param conf_thres: int
        この閾値以上のスコアを持つ物体検出結果のみ処理の対象にする
    :param iou_thres: int
        boxes 交差率（重なり具合）が閾値より高い場合、スコアの高い方を選定
    :param agnostic: bool
        Trueで class に関係なく処理を実行
    :param max_det: int
        NMS後に残す物体検出結果の最大数
    :param nm: int
        num_classes

    :return: List[torch.Tensor]
        [ num_boxes, 6 + num_masks ][x1, y1, x2, y2, confidence, class, mask1, mask2, ...]
    """

    batch_size = prediction.shape[0]
    class_num = prediction.shape[1] - nm - 4
    # masks が始まるインデックス
    mi = 4 + class_num

    # 各 class における検出スコアの最大値が conf_thres 以上のものを抽出
    xc = prediction[:, 4:mi].amax(1) > conf_thres

    # NMS処理における最大 box_size
    max_wh = 768
    # NMS処理における最大処理件数
    max_nms = 30000

    # 出力tensor
    output = [torch.zeros((0, 6 + nm))] * batch_size

    # confidence
    x = prediction[0].transpose(0, -1)[xc[0]]

    # 検出結果がなければ空のtensorを返す
    if not x.shape[0]:
        return output

    # bboxes、classes、masks をそれぞれ別のtensorに分割
    box, cls, mask = x.split((4, class_num, nm), 1)

    # xywh座標をxyxy座標に変換
    box = xywh2xyxy(box)

    # 各検出結果における class ラベルと最大スコアを取得
    conf, j = cls.max(1, keepdim=True)

    # スコア閾値以上の検出結果のみを残し、tensorを結合
    x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

    # 検出結果の個数
    n = x.shape[0]
    if not n:
        return output

    # スコアの高い順に max_nms 個の検出結果を残す
    x = x[x[:, 4].argsort(descending=True)[:max_nms]]

    # box_size を計算
    c = x[:, 5:6] * (0 if agnostic else max_wh)
    boxes, scores = x[:, :4] + c, x[:, 4]

    # NMS処理の実行
    i = torchvision.ops.nms(boxes, scores, iou_thres)
    i = i[:max_det]

    # batch_size が 1 前提の処理なので、2以上の場合は書き換えの必要あり
    output[0] = x[i]

    return output



def xywh2xyxy(x):

    """
    座標の変換
    参考 :
    https://github.com/ultralytics/ultralytics/blob/d608565a17384fe8bd666bd91318191846b50cbc/ultralytics/utils/ops.py#L402

    :param x: torch.Tensor
        xywh 形式 (x_center, y_center, width, height)
    :return: torch.Tensor
        xyxy 形式 (x_min, y_min, x_max, y_max)
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y

    return y



def process_mask(protos, masks_in, bboxes, shape, upsample=True):

    """
    bounding box の結果に mask を適応する
    参考 :
    https://github.com/ultralytics/ultralytics/blob/d608565a17384fe8bd666bd91318191846b50cbc/ultralytics/utils/ops.py#L670

    :param protos: torch.Tensor[mask_dim, mask_h, mask_w]
        推論結果の mask (32*160*160)
    :param masks_in: torch.Tensor[n, mask_dim]
        NMS処理後の mask , n はNMS処理後の mask 数
    :param bboxes: torch.Tensor[n, 4]
        bounding boxes　xyxy座標(x1, y1, x2, y2) , n はNMS後の mask 数
    :param shape: tuple
        CameraX解析画像のサイズ　640,640
    :param upsample: bool
        Trueで、mask を shape までアップサンプル

    :return: torch.Tensor[n, h, w]
        n はNMS後の mask 数
        入力画像の高さと幅
    """

    c, mask_h, mask_w = protos.shape
    image_height, image_width = shape

    masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mask_h, mask_w)

    # 検出された bboxes は入力画像のスケールで表現されているため、
    #  mask_h, mask_w のスケールに合わせるために幅を縮小率で割る
    downsampled_bboxes = bboxes.clone()
    downsampled_bboxes[:, 0] *= mask_w / image_width
    downsampled_bboxes[:, 2] *= mask_w / image_width
    downsampled_bboxes[:, 3] *= mask_h / image_height
    downsampled_bboxes[:, 1] *= mask_h / image_height

    # mask を bounding box に切り抜く
    masks = crop_mask(masks, downsampled_bboxes)

    if upsample:
        masks = F.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]

    return masks.gt_(0.5)



def crop_mask(masks, boxes):

    """
    mask と bounding box を受け取り、bounding box に切り抜かれた mask を返す
    n は mask の数
    参考 :
    https://github.com/ultralytics/ultralytics/blob/d608565a17384fe8bd666bd91318191846b50cbc/ultralytics/utils/ops.py#L630

    :param masks: torch.Tensor[n, h, w]
    :param boxes: torch.Tensor[n, 4]

    :return: torch.Tensor
    """

    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))



def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):

    """
    img1_shape の座標を img0_shape にスケーリング
    (640,640) -> (640, 480)　に変換
    参考 :
    https://github.com/ultralytics/ultralytics/blob/d608565a17384fe8bd666bd91318191846b50cbc/ultralytics/utils/ops.py#L88

    :param img1_shape: tuple
        グレーでパディング済みの解析画像 (640, 640)
    :param boxes: torch.Tensor
        bbox xyxy座標
    :param img0_shape: tuple
        ターゲットサイズ (640, 480)
    :param ratio_pad: tuple
        元画像サイズとパディング後の画像サイズの比率を表す (ratio, pad)
    :return: torch.Tensor
        スケーリング後 xyxy座標
    """

    # パディングとして差し引く値を計算
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]
    boxes[..., [1, 3]] -= pad[1]
    boxes[..., :4] /= gain

    # 境界内に確実に収まるように切り抜く
    return clip_boxes(boxes, img0_shape)



def clip_boxes(boxes, shape):

    """
    bbox が shape に収まるように切り抜く
    参考 :
    https://github.com/ultralytics/ultralytics/blob/d608565a17384fe8bd666bd91318191846b50cbc/ultralytics/utils/ops.py#L305

    :param boxes: torch.Tensor
        bbox
    :param shape: tuple
        切り抜きサイズ

    :return: torch.Tensor
        切り抜き後
    """

    if isinstance(boxes, torch.Tensor):
        boxes[..., 0].clamp_(0, shape[1])
        boxes[..., 1].clamp_(0, shape[0])
        boxes[..., 2].clamp_(0, shape[1])
        boxes[..., 3].clamp_(0, shape[0])
    else:
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])

    return boxes



def calc_ratio(position, mpitchx):

    """
    デバイスの傾きとディスプレイ上のグラスの位置から、倍率を計算

    :param position: int
        検出したオブジェクトのy座標
    :param mpitchx: int
        デバイスの傾き

    :return: int / float
        if 0 < output < 0.85 then output
        elif output < 0 then 0
        else then 9999
    """

    if -10 <= mpitchx <=0:
        min_r = 0.000006155020573 * (position ** 2) - 0.004117655571876 * position + 0.681662839134254
        max_r = 0.000000000256780 * (position ** 4) - 0.000000458639530 * (position ** 3) + 0.000300937105127 * (position ** 2) - 0.084022660342011 * position + 8.411846952391370

        r_ratio = min_r * (abs(mpitchx)/10) + max_r * (1-abs(mpitchx)/10)

    elif mpitchx <= 10:
        min_r = 0.000000000256780 * (position ** 4) - 0.000000458639530 * (position ** 3) + 0.000300937105127 * (position ** 2) - 0.084022660342011 * position + 8.411846952391370
        max_r = -0.000001793716450 * (position ** 2) + 0.003472521645861 * position - 0.722665678971605

        r_ratio = min_r * (1-mpitchx/10) + max_r * (mpitchx/10)

    elif mpitchx <= 20:
        min_r = -0.000001793716450 * (position ** 2) + 0.003472521645861 * position - 0.722665678971605
        max_r = 0.000000009088620 * (position ** 3) - 0.000010690532181 * (position ** 2) + 0.005626563254962 * position - 0.652017152503150

        r_ratio = min_r * (1-(mpitchx-10)/10) + max_r * ((mpitchx-10)/10)

    elif mpitchx <= 30:
        min_r = 0.000000009088620 * (position ** 3) - 0.000010690532181 * (position ** 2) + 0.005626563254962 * position - 0.652017152503150
        max_r = -0.000000233085162 * (position ** 2) + 0.001999609088435 * position - 0.151016250624041

        r_ratio = min_r * (1-(mpitchx-20)/10) + max_r * ((mpitchx-20)/10)

    elif mpitchx <= 40:
        min_r = -0.000000233085162 * (position ** 2) + 0.001999609088435 * position - 0.151016250624041
        max_r = 0.000001343311718 * (position ** 2) + 0.001125375259159 * position + 0.127390261205400

        r_ratio = min_r * (1-(mpitchx-30)/10) + max_r * ((mpitchx-30)/10)

    elif mpitchx <= 50:
        min_r = 0.000001343311718 * (position ** 2) + 0.001125375259159 * position + 0.127390261205400
        max_r = -0.000000137831411 * (position ** 2) + 0.001476858946684 * position + 0.329448764745298

        r_ratio = min_r * (1-(mpitchx-40)/10) + max_r * ((mpitchx-40)/10)

    elif mpitchx <= 60:
        min_r = -0.000000137831411 * (position ** 2) + 0.001476858946684 * position + 0.329448764745298
        max_r = -0.000000580856986 * (position ** 2) + 0.001323320826443 * position + 0.452189845033462

        r_ratio = min_r * (1-(mpitchx-50)/10) + max_r * ((mpitchx-50)/10)

    elif mpitchx <= 70:
        min_r = -0.000000580856986 * (position ** 2) + 0.001323320826443 * position + 0.452189845033462
        max_r = -0.000005722691815 * (position ** 2) + 0.002124216034722 * position + 0.619285916540132

        r_ratio = min_r * (1-(mpitchx-60)/10) + max_r * ((mpitchx-60)/10)

    else:
        return 9999

    if r_ratio<0:
        return 0
    elif r_ratio<0.85:
        return float(r_ratio)
    else:
        return 9999