import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt


def calculateDistance(x,y):
    if len(x) != len(y):
        return -1
    sum = 0.0
    for i in range(len(x)):
        diff = x[i]-y[i]
        sum=sum+diff*diff
    sum = math.sqrt(sum)
    return sum


def captureHSVHistogramOfROI(frame, UppLeftCornerPos,LowRightCornerPos):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    roi = hsv[UppLeftCornerPos[1]:LowRightCornerPos[1], UppLeftCornerPos[0]:LowRightCornerPos[0]]
    hist_mask = cv.inRange(roi, (0, 1, 13), (180, 255, 255))

    roi_hist = cv.calcHist([roi], [0, 1], hist_mask, [9,12], [0, 180, 0, 255])
    cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
    cv.imshow('roi_hist', roi_hist)
    return roi_hist

def showHistogramOfROI(frame, UppLeftCornerPos,LowRightCornerPos):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    roi = hsv[UppLeftCornerPos[1]:LowRightCornerPos[1], UppLeftCornerPos[0]:LowRightCornerPos[0]]

    color = ('b', 'g', 'r')
    titles = ('h', 's', 'v')
    fig, axs = plt.subplots(3)
    for i, col in enumerate(color):
        histr = cv.calcHist([roi], [i], None, [256], [1, 256])
        cv.normalize(histr, histr)
        axs[i].plot(histr, color=col)
        axs[i].set_title(titles[i])
        axs[i].set(xlim=(0, 255))
    plt.show()

#zwraca obraz po nalozeniu maski odslaniajaca tylko obszary o kolorze skory
def colorSegmentationInHSV(frame):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    mask_low = cv.inRange(hsv, (0, 30, 0), (18, 160, 255))
    mask_high = cv.inRange(hsv, (140, 12, 0), (180, 140, 255))
    mask_medium = cv.inRange(hsv, (130, 12, 100), (140, 70, 210))


    mask_lowlight_combined = cv.bitwise_or(mask_low, mask_high)
    mask_highlight_combined = cv.bitwise_or(mask_medium, mask_medium)
    mask_combined = cv.bitwise_or(mask_lowlight_combined, mask_highlight_combined)

    frame_masked = cv.bitwise_and(frame,frame,mask=mask_combined)
    return frame_masked

def segmentColorByBackprojection(frame,hand_hist):
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    object_segment = cv.calcBackProject([hsv_frame], [0, 1], hand_hist, [0, 255, 0, 255], 1)
    return object_segment


#uchwyc statyczne tlo
def catchStaticBackground():
    global bgModel
    bgModel = cv.createBackgroundSubtractorMOG2(history=10, varThreshold=30, detectShadows=False)

def maskStaticBackground(frame,static_background,threshold = 5):
    frame_gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    diff = cv.absdiff(frame_gray,static_background)
    ret, frame_thresh = cv.threshold(diff, threshold, 255, cv.THRESH_BINARY)
    frame = cv.bitwise_and(frame,frame,mask = frame_thresh)
    return frame


bgModel = None

circle_rad=0
showRectangle = False
segmentationByBackprojection = False
histOfROICaptured = False

# prostokat do pobrania histogramu koloru skory
rectanglePos0 = (150, 150)
rectangleSize = (100,150)
rectanglePos1 = (rectanglePos0[0] + rectangleSize[0],rectanglePos0[1] + rectangleSize[1])

x,y,w,h=(0,0,0,0)
srodek_okregu = [0,0]
iteratorek = 0
zmianyLiczbyPalcow = 0
poprzedniaLiczbaPalcow = 0
it_usunRysunek=0

cap = cv.VideoCapture(0)

#wylaczenie automatycznych ustawien kamery, ustawienie czasu ekspozycji
cap.set(cv.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 0)
cap.set(cv.CAP_PROP_EXPOSURE, -5.2)

static_background = catchStaticBackground()

if cap.isOpened():
    while (True):

        ret, frame = cap.read()
        if not ret:
            print("Nie udało się uchwycic obrazu. Koniec programu")
            break

        # lustrzane odbicie, zeby wyswietlac po ludzku
        frame = cv.flip(frame, 1)
        # zapis ramki do wyswietlania
        frame_toshow = frame.copy()
        # nalozenie maski zerujacej statyczne tlo
        # learningRate=0, zeby tlo nie zmienialo sie, bo reka bedzie zanikac
        # erozja, zeby pozbyc sie szumu z drgan laptopa, czestotliwosci oswietlenia i innych
        fgmask_staticbckground = bgModel.apply(frame, learningRate=0)
        fgmask_staticbckground = cv.erode(fgmask_staticbckground, np.ones((3, 3), np.uint8))
        without_staticbckground = cv.bitwise_and(frame, frame, mask=fgmask_staticbckground)
        # zamkniecie, zeby usunac falszywe czarne piksele
        # *widoczne przy dobrze oswietlonym tle i ostrych ksztaltach w tle
        without_staticbckground = cv.morphologyEx(without_staticbckground, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (7, 7)))

        # segmentacja ze wzgledu na kolor, zeby zapobiec sytuacji, gdy maska nie zakryje tla
        # np. ruch laptopa, poruszenie twarza, reka z bluza itp.
        # pozostaja tylko poruszajace sie elementy o kolorze skory
        f_after_colormask = colorSegmentationInHSV(without_staticbckground)


        # zmiana bgr2gray, , zeby zastosowac operacji morfologiczne
        # gray zamiast obraz binarny
        gray = cv.cvtColor(f_after_colormask, cv.COLOR_BGR2GRAY)
        opening = cv.morphologyEx(gray, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7)))
        closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (7, 7)), iterations = 2)
        # filtr, zby pozbyc sie zaklocen i rozmyc ostry ksztalt po poprzednich operacjach
        blurred = cv.filter2D(closing, -1, np.ones((9, 9), np.float32) / 81)

        # binaryzacja i znow operacje morfologiczne
        # pozbycie sie nadmiarowych bialych pikseli i dziur
        # celem jest utrzymac przede wszystkim mocny, stabilny kontur
        ret, frame_thresh = cv.threshold(gray, 20, 255, cv.THRESH_BINARY)
        frame_thresh = cv.erode(frame_thresh, np.ones((2, 2), np.uint8), iterations=3)
        frame_thresh = cv.dilate(frame_thresh, np.ones((3, 3), np.uint8), iterations=2)


        frame_filled_contour = np.zeros_like(frame_thresh)
        # wykryj kontury
        contours, hierarchy = cv.findContours(frame_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if len(contours)!=0:
            # wez kontur o najwiekszym polu
            cnt = max(contours, key = cv.contourArea)
            # rysowanie najwiekszego konturu dla zobrazowania
            cv.drawContours(frame,[cnt],0,(0,255,0),-1)
            # narysuj kontur wypelniony wewnatrz, aby pozbyc sie ewentualnych dziur wewnatrz
            # lepsze wyjscie, poniewaz operacje morfologiczne poza lataniem dziur 'sklejaja' palce
            cv.drawContours(frame_filled_contour, [cnt], 0, (255, 255, 255), -1)
            # prostokat okrywajacy kontur
            x,y,w,h = cv.boundingRect(cnt)
            # narysuj prostokat okrywajacy kontur dla zobrazowania
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

        # morfologiczne zamkniecie, zeby pozbyc sie ewentualnych dziur w konturze
        frame_filled_contour = cv.morphologyEx(frame_filled_contour, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
        cv.imshow('frame_with_biggest contour',frame_filled_contour)
        # wyliczenie dystansu do czarnych pikseli dla kazdego punktu na obrazie
        # wybranie punktu o najwiekszej odleglosci jako srodek, a dystansu jako promien
        # okregu wpisanego we wnetrze dloni
        dist_img = cv.distanceTransform(frame_filled_contour,cv.DIST_L1,cv.DIST_MASK_3)
        circle_cen = np.unravel_index(np.argmax(dist_img, axis=None), dist_img.shape)
        circle_rad = int(0.5*dist_img[circle_cen[0]][circle_cen[1]]+0.5*circle_rad)
        # narysuj okrag wpisany w dlon
        cv.circle(frame, (circle_cen[1], circle_cen[0]), circle_rad, (255, 255, 0), 6)


        # wspolrzedna dolnej czesci dloni wyznaczajaca obszar,
        # ponizej ktorego nie ma juz dloni, a zaczyna sie nadgarstek
        Ydown_frame = int(circle_cen[0]+circle_rad)

        # tworzenie maski, ktora z obrazu wytnie tylko dlon
        # wyciecie w postaci prostokatu o wierzcholkach
        # gorny prawy tak, jak prostokat obejmujacy caly kontur dloni z reka
        # dolny lewy punkt w osi OX tak jak prostokąt obejmujacy caly kontur dloni
        # w osi OY o promien okregu wpisanego w dlon nizej niz centrum tego okregu
        mask_nothand = np.zeros_like(frame_thresh)
        y1 = max(y , Ydown_frame)
        cv.rectangle(mask_nothand, (x, y), (x + w, y1), (255, 255, 255), -1)

        # wyciecie dloni przy pomocy maski
        frame_palm_only=cv.bitwise_and(frame_filled_contour, frame_filled_contour, mask=mask_nothand)

        cv.imshow('frame_palm_only', frame_palm_only)


        contours, hierarchy = cv.findContours(frame_palm_only, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if len(contours) != 0:
            cnt = max(contours, key=cv.contourArea)


            hull = cv.convexHull(cnt,returnPoints=False)
            defects = cv.convexityDefects(cnt, hull)

            ## PROG WYKRYCIA CZUBKA PALCA
            # fioletowy okrag na obrazie frame
            cv.circle(frame, (circle_cen[1], circle_cen[0]), int(1.6*circle_rad), (255, 0, 255), 6)
            cv.circle(frame, (circle_cen[1], circle_cen[0]), 7, (255, 0, 255), -1)
            if defects is None:
                print("defect none")
            else:
                count_holes = 0
                count_tipsOfFingers = 0
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])
                    cv.line(frame, start, end, [0, 255, 0], 2)
                    cv.circle(frame, far, 5, [0, 255, 255], -1)
                    cv.circle(frame, start, 5, [0, 0, 0], -1)

                    if d/256.0 > 0.6*circle_rad :
                        count_holes = count_holes + 1
                        ## rozpoznawanie liczby palcow po liczbach wypuklych punktow
                        # nie dziala, bo ppunkty startowe defektow wystepuja po kilka na jednym palcu
                        # sposob: liczaba palcow = count_holes +1
                        # gdy liczba palcow = 1, to sprawdzic, czy istnieje punkt start powyzej circle_rad*1.6/1.7
                        # tak odrozniamy 1 palec od piesci lub pustej dloni
                    if calculateDistance(start,(circle_cen[1], circle_cen[0])) > circle_rad *1.6:
                        count_tipsOfFingers = count_tipsOfFingers + 1
                    no_of_fingers = count_holes + 1
                    if count_tipsOfFingers == 0:
                        no_of_fingers = 0

                    cv.putText(frame, str(no_of_fingers), (100, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv.LINE_AA)


                    ## ogarnac temat rysowania, bo to jakis koszmar xD
                    # zaczyna punkt w srodku i przemieszczanie inkrementacyjne
                    if 'rysowanie' not in locals():
                        rysowanie = np.zeros_like(frame)
                    cv.rectangle(rysowanie, (50, 25), (70, 55), (0, 0, 0), -1)
                    cv.putText(rysowanie, str(no_of_fingers), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                               cv.LINE_AA)

                    iteratorek = iteratorek + 1
                    zmianyLiczbyPalcow = zmianyLiczbyPalcow + no_of_fingers - poprzedniaLiczbaPalcow

                    if iteratorek > 3:
                        if zmianyLiczbyPalcow == 0:
                            # ruchoma srednia
                            srodek_okregu = [int(0.3*circle_cen[1] +0.7*srodek_okregu[0]), int(0.3*circle_cen[0] + 0.7*srodek_okregu[1])]

                            if no_of_fingers == 0:
                                it_usunRysunek = it_usunRysunek+1
                                if it_usunRysunek > 5:
                                    rysowanie = np.zeros_like(frame)
                                    it_usunRysunek=0
                            elif no_of_fingers == 1:
                                rysowanie_kolor = (255,0,0)
                            elif no_of_fingers == 2:
                                rysowanie_kolor = (255,255,0)
                            elif no_of_fingers == 3:
                                rysowanie_kolor = (0, 255, 0)
                            elif no_of_fingers == 4:
                                rysowanie_kolor = (0, 255, 255)
                            elif no_of_fingers == 5:
                                rysowanie_kolor = (0, 0, 255)
                            else:
                                rysowanie_kolor = (0, 0, 0)

                            if no_of_fingers != 0:
                                cv.circle(rysowanie, tuple(srodek_okregu), 3, rysowanie_kolor, -1)
                            cv.imshow('rysowanie', rysowanie)
                        iteratorek = 0
                        zmianyLiczbyPalcow = 0
                        poprzedniaLiczbaPalcow = no_of_fingers


        key = cv.waitKey(10)

        # segmentacja na podstawie wstecznej projekcji histogramu
        # jako ciekawostka (lepiej miec niz nie miec)
        if segmentationByBackprojection and histOfROICaptured:
            backprojection_f = segmentColorByBackprojection(without_staticbckground, hand_hist)
            masked_backproject = cv.inRange(backprojection_f, 8, 255)
            backproject_mask_dilated = cv.dilate(masked_backproject,np.array((3,3),np.uint8),iterations = 2)
            cv.imshow('backproject_mask_dilated', backproject_mask_dilated)

        # podswietlenie prostokata do wyznaczenia histogramu reki
        if showRectangle:
            cv.rectangle(frame,rectanglePos0,rectanglePos1,(255, 255, 0),5)

        cv.imshow('frame_thresh', frame_thresh)
        cv.imshow('frame', frame)
        cv.imshow('clean_frame',frame_toshow)
        cv.imshow('no_static_backgroung', without_staticbckground)


        if key == ord('i') and showRectangle:
                hand_hist = captureHSVHistogramOfROI(without_staticbckground,rectanglePos0,rectanglePos1)
                histOfROICaptured = True
        if key == ord('o'):
            segmentationByBackprojection = not segmentationByBackprojection
        if key == ord('u'):
            showRectangle = not showRectangle
        if key == ord('k'):
            showHistogramOfROI(frame,rectanglePos0,rectanglePos1)
        if key == ord('m'):
            static_background = catchStaticBackground()

        if key == 27 or key == ord('q'):
            break
else:
    print("Nie udalo sie otworzyc kamery.")

cv.destroyAllWindows()
cap.release()
