import cv2
import numpy as np

filename = "input.jpg"  # 입력 이미지 파일명을 적으세요.
img = cv2.imread(filename, cv2.IMREAD_COLOR)
# -----------------------------------------------
# 공통 작업 : 이미지 열기 / 크롭 / hsv 추출

# 1) 이미지 카피 / 리사이즈
img_copy = img.copy()
img_resize = cv2.resize(img_copy, (500, 500))
img_resize1 = img_resize.copy()
img_resize2 = img_resize.copy()

# 2) 전체 펜 뚜껑 크롭
pen_cover = img_resize[60:135, :]

# 3) 중앙에 있는 글자(네임펜 마크) : 크롭
center_letter = img_resize[180:240, :]
lower_letter = img_resize[270:320, :]

# 4) 중앙 글자(네임펜 마크) : bgr > hsv
letter_hsv = cv2.cvtColor(center_letter, cv2.COLOR_BGR2HSV)
letter_hsv2 = cv2.cvtColor(lower_letter, cv2.COLOR_BGR2HSV)

# 5) 중앙 글자(네임펜 마크) hsv 이미지 > 노란색만 보이게 하기, 나머지는 검정색 처리
# 픽셀 범위 지정
lower_Y = np.array([10, 200, 100])
upper_Y = np.array([45, 255, 255])
lower_Y2 = np.array([10, 100, 100])

# 노란 부분만 표시
mask = cv2.inRange(letter_hsv, lower_Y, upper_Y)
letter_yellow = cv2.bitwise_and(center_letter, center_letter, mask=mask)

mask2 = cv2.inRange(letter_hsv2, lower_Y2, upper_Y)
l_letter_yellow = cv2.bitwise_and(lower_letter, lower_letter, mask=mask2)

# 6) hsv 분리
h, s, v = cv2.split(letter_yellow)
h2, s2, v2 = cv2.split(l_letter_yellow)

# ---------------------------------------------
# R1 : 색 위치 판단

# 1) 위치 선정
# 중앙 글자(네임펜 마크) > 노란색 시작부분(s), 타원의 끝부분(e) 좌표값 알아내기
for x in range(0, 500):
    for y in range(0, 60):
        if h[y][x] > 0:
            e = x
for x in range(0, 500):
    for y in range(0, 60):
        if h[y][499 - x] > 0:
            s = 499 - x

# 2) 펜 뚜껑 위치 정밀조정
# pen_width = 펜 하나의 가로 넓이
# cell_width = 펜 하나를 가로로 3등분한 넓이
pen_width = (e - s) / 7
cell_width = (e - s) / 21

# btw_sx : s지점과 가까운 펜과 펜사이의 중간에 있는 빈공간 x좌표
btw_sx = round((10 / 43) * pen_width + s)
# first_pen : 첫번째 펜에서 뚜껑부분을 크롭할 때, 시작점 x좌표
first_pen = round(btw_sx - (cell_width * 11))

# 3) 배열선언
# 정답의 hsv값 최대, 최소 배열
answer_array = [[19, 160, 3, 176, 90, 75, 103, 113, 118, 14, 0, 106],
                [25, 162, 4, 177, 91, 77, 104, 115, 120, 18, 179, 113],
                [117, 103, 160, 156, 179, 202, 197, 168, 128, 0, 52, 44],
                [211, 130, 181, 208, 252, 223, 207, 215, 151, 155, 128, 125],
                [177, 170, 205, 188, 88, 164, 226, 150, 172, 0, 82, 29],
                [226, 192, 222, 239, 127, 178, 235, 244, 202, 180, 140, 115]]
# 시각화를 위한 RGB 배열 선언
color_i = [(51, 183, 217), (151, 103, 185), (71, 86, 211), (66, 47, 203), (100, 99, 11), (102, 170, 28),
           (230, 149, 47), (203, 74, 40), (180, 83, 79), (104, 130, 154), (69, 66, 97), (55, 40, 33)]

# 정답의 hsv와 테스트이미지의 hsv를 비교 후, 정오 판단을 넣을 배열
test_array = [[], [], []]
# txt 성능평과 결과를 위한 배열 선언
pred_R1 = []
# result에 들어가는 좌표값을 넣어줄 배열(커버 x좌표값[시작부분][끝부분])
result_in = []

# 4) 비교
for i in range(0, 12):

    # <좌표>
    # cover_start : 뚜껑에서 크롭할 부분의 시작점 x좌표
    # cover_last : 뚜껑에서 크롭할 부분의 끝나는 점 x좌표
    cover_start = first_pen + (round(pen_width * i))
    cover_last = cover_start + round(cell_width)

    # 모든 펜의 뚜껑이 보이도록 잘라낸 부분 중에서, result = 하나의 펜 뚜껑 색만 보이도록 자르겠다.
    result = pen_cover[:, cover_start:cover_last]

    # (R2에서 사용 : 측정 좌표값 result_in배열에 저장 (커버 x좌표값[시작부분][끝부분]))
    result_in.append([cover_start, cover_last])

    # <HSV>
    # 자른 부분(result)의 hsv값을 추출함
    result = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(result)
    # 자른부분 hsv에서 각각의 평균값
    result_h = int(np.mean(h))
    result_s = int(np.mean(s))
    result_v = int(np.mean(v))

    # <비교>
    if answer_array[0][i] - 5 <= result_h <= answer_array[1][i] + 5:
        test_array[0].append(1)
    else:
        test_array[0].append(0)

    if answer_array[2][i] - 5 <= result_s <= answer_array[3][i] + 5:
        test_array[1].append(1)
    else:
        test_array[1].append(0)

    if answer_array[4][i] - 5 <= result_v <= answer_array[5][i] + 5:
        test_array[2].append(1)
    else:
        test_array[2].append(0)

    # <판단>
    # 일치
    if test_array[0][i] == 1 and test_array[1][i] == 1 and test_array[0][i] == 1:
        # 갈색
        if i == 10 and np.min(h) == 0 and np.max(h) == 179:
            pred_R1.append(i + 1)

        # 11번째 자리 불일치
        elif i == 10 and np.min(h) != 0 and np.max(h) != 179:
            # 시각화에 쓰일 숫자 선언
            number = "0"

            for j in range(0, 12):
                # 파랑 & 검정
                if 106 <= result_h <= 115:
                    if 160 <= result_s <= 255:
                        number = str(8)
                        break
                    else:
                        number = str(12)
                        break

                # h로 색 찾기
                elif answer_array[1][j] - answer_array[0][j] <= 5 \
                        and answer_array[0][j] <= result_h <= answer_array[1][j]:
                    number = str(j + 1)
                    break

                # hsv로 색 찾기
                elif answer_array[0][j] - 5 <= result_h <= answer_array[1][j] + 5 \
                        and answer_array[2][j] - 5 <= result_s <= answer_array[3][j] + 5 \
                        and answer_array[4][j] - 5 <= result_v <= answer_array[5][j] + 5:
                    number = str(j + 1)
                    break

            # 오답 시각화 <2> : 정답의 색깔 표시
            img_vis1 = cv2.rectangle(img_resize, (cover_start - 5, 72), (cover_last, 150), (0, 255, 0), 3)

            img_vis1 = cv2.rectangle(img_resize, (cover_start - 2, 75), (cover_last - 3, 148), color_i[i], -1)
            img_vis1 = cv2.arrowedLine(img_resize, (cover_start + 5, 79), (cover_start + 5, 62), (255, 255, 255),
                                               thickness=2, tipLength=0.3)

            # 오답 시각화 <2> : 숫자 표시
            img_vis1 = cv2.rectangle(img_resize, (cover_start - 5, 170), (cover_last + 13, 210), (255, 255, 255), -1)
            img_vis1 = cv2.putText(img_resize, number, (cover_start - 8, 200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                    (0, 0, 255), 2, cv2.LINE_AA)

            # txt 성능평가 결과
            pred_R1.append(int(number))

        # 일치
        else:
            pred_R1.append(i + 1)

    # 색 불일치
    else:
        # 시각화에 쓰일 숫자 선언
        number = "0"

        for j in range(0, 12):
            # 파랑 & 검정
            if 106 <= result_h <= 115:
                if 160 <= result_s <= 255:
                    number = str(8)
                    break
                else:
                    number = str(12)
                    break

            # h로 색 찾기
            elif answer_array[1][j] - answer_array[0][j] <= 5 \
                    and answer_array[0][j] <= result_h <= answer_array[1][j]:
                number = str(j + 1)
                break

            # hsv로 색 찾기
            elif answer_array[0][j] - 5 <= result_h <= answer_array[1][j] + 5 \
                    and answer_array[2][j] - 5 <= result_s <= answer_array[3][j] + 5 \
                    and answer_array[4][j] - 5 <= result_v <= answer_array[5][j] + 5:
                number = str(j + 1)
                break

        # 오답 시각화 <2> : 정답의 색깔 표시
        img_vis1 = cv2.rectangle(img_resize, (cover_start - 5, 72), (cover_last, 150), (0, 255, 0), 3)

        img_vis1 = cv2.rectangle(img_resize, (cover_start - 2, 75), (cover_last - 3, 148), color_i[i], -1)
        img_vis1 = cv2.arrowedLine(img_resize, (cover_start + 5, 79), (cover_start + 5, 62), (255, 255, 255),
                                   thickness=2, tipLength=0.3)

        # 오답 시각화 <2> : 숫자 표시
        img_vis1 = cv2.rectangle(img_resize, (cover_start - 5, 170), (cover_last + 13, 210), (255, 255, 255), -1)
        img_vis1 = cv2.putText(img_resize, number, (cover_start - 8, 200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                               (0, 0, 255), 2, cv2.LINE_AA)

        # txt 성능평가 결과
        pred_R1.append(int(number))

# R1이 모두 통과되었을 경우 시각화
if pred_R1 == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
    img_vis1 = cv2.putText(img_resize, "R1_PERFECT!!", (150, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 3,
                           cv2.LINE_AA)

# ---------------------------------------------
# R2 : 펜 몸통 정렬 판단

# 1) 알고리즘 방향
# 정답 : 각각의 펜에서 노란색의 네임펜 마크 인식이 되면 바르게 정렬 된 것
# 오답 : 노란색 네임펜 마크가 인식이 안되는 경우

# 1-1) 각각의 펜에서 "nam"부분, h값을 배열에 저장
addcolor_on = []
for x in range(0, 500):
    addcolor_in = []
    for y in range(0, 50):
        if h2[y][x] > 0:
            addcolor_in.append(h2[y][x])
    addcolor_on.append(sum(addcolor_in))

# 1-2) 노란색 마크가 인식이 되면, addcolor_fin의 값이 100이상이다.
# 100이상일때 x 좌표값을 배열에 저장
addcolor_fin = []
for i in range(0, 500):
    if addcolor_on[i] > 100:
        addcolor_fin.append(i)
addcolor_fin.append(0)

# 1-3) 마크가 바르게 정렬된 펜의 좌표를 확인하는 과정
a = 0
a1 = 0
addcolor_length_boundary = []
for x in range(0, len(addcolor_fin) - 1):
    if addcolor_fin[x] + 1 == addcolor_fin[x + 1]:
        a = a + 1
    else:
        if a >= 9:
            a1 = a1 + 1
            addcolor_length_boundary.append([addcolor_fin[x] - a, addcolor_fin[x] - a / 2, addcolor_fin[x], a + 1])
        a = 0

# 2) 판단
# 2-1) 정렬 잘됨
# 펜 뚜껑 좌표값(R1)을 이용하여 노란 펜 번호 저장 & 시각적으로 보이기
# 뚜껑 좌표값+-10 안에 [중간값]이 있을경우 펜 번호 저장
right_pen = []
for a2 in range(0, 12):
    for a3 in range(0, a1):
        if (result_in[a2][0] - 10) <= addcolor_length_boundary[a3][1] <= (result_in[a2][1] + 10):
            right_pen.append(a2)

            # 정답 시각화
            img_vis2 = cv2.putText(img_resize2, "o", (addcolor_length_boundary[a3][0] - 5, 280),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 4, cv2.LINE_AA)

# 2-2) 정렬 잘 안됨
# 펜의 번호
list_all = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# txt 성능평가 결과 도출을 위한 배열
pred_R2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# 모든 펜의 번호 중 정렬이 잘 된 펜의 번호를 지움
for i in range(0, 12):
    for j in range(0, len(right_pen)):
        if i == right_pen[j]:
            list_all.remove(i)

            # txt 성능평가
            pred_R2[i] = 1

# 오답 시각화 : 빨간 네모상자 표시
for a6 in range(0, len(list_all)):
    a7 = list_all[a6]
    if a7 < 3:
        move_r = 7
        img_vis2 = cv2.rectangle(img_resize2, (result_in[a7][0] + move_r, 250), (result_in[a7][1] + move_r, 500),
                                 (0, 0, 255), 3)
    elif 3 <= a7 <= 8:
        img_vis2 = cv2.rectangle(img_resize2, (result_in[a7][0], 250), (result_in[a7][1], 500),
                                 (0, 0, 255), 3)
    elif a7 > 8:
        move_l = -7
        img_vis2 = cv2.rectangle(img_resize2, (result_in[a7][0] + move_l, 250), (result_in[a7][1] + move_l, 500),
                                 (0, 0, 255), 3)

# ========알고리즘 및 시각화 소스코드 작성 (끝)=========
#  예측 결과 txt로 제출(성능평가에 사용, 이 부분은 수정하지 마시오)
txt_name = str('answer(') + filename + str(').txt')
f = open(txt_name, mode='wt', encoding='utf-8')
for i in range(12):
    f.write('%s\n' % str(pred_R1[i]))
for j in range(12):
    f.write('%s\n' % str(pred_R2[j]))
f.close()

# 시각화 결과 표시(예측 결과 확인용, 이 부분은 수정하지 마시오)
cv2.imshow('visualization_R1', img_vis1)  # 시각화(R1)
cv2.imshow('visualization_R2', img_vis2)  # 시각화(R2)
cv2.waitKey(0)
cv2.destroyAllWindows()
