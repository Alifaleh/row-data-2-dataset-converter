welcome = '''
########################################################################################################################
###                                      welcome to row data 2 datet converter                                     ###
#####                                                                                                              #####
#######                                             BY:Ali A.Falih                                               #######
#####                                                                                                              #####
###                                          Email:Alifalih783783@gmail.com                                          ###
########################################################################################################################
'''
print(welcome)
print("\n")
import numpy as np
import pandas as pd
from scipy.fftpack import rfft, irfft, fftfreq
import matplotlib.pyplot as plt
import pywt
import math
import os
import re


def get_data():
    pl = []
    csvs = []
    digs = []

    for n in os.listdir():
        if os.path.isdir(n):
            pl.append(n)
    print(pl)
    root = os.getcwd()

    for p in pl:
        os.chdir(root + "\\" + p)
        pcsv = []
        phea = []
        pdig = []
        pf = os.listdir()
        for f in pf:
            if f[-3:] == "csv" or f[-3:] == "CSV":
                if f[:-4] + ".dat" in pf or f[:-4] + ".DAT" in pf:
                    if f[:-4] + ".hea" in pf or f[:-4] + ".HEA" in pf:
                        pcsv.append(f)
                        if f[:-4] + ".hea" in pf:
                            phea.append(f[:-4] + ".hea")
                        elif f[:-4] + ".HEA" in pf:
                            phea.append(f[:-4] + ".hea")
        for h in phea:
            file = open(root + "\\" + p + "\\" + h)
            d = file.read()
            d = re.findall("#Diagnosis report:.*", d)[0][19:]
            pdig.append(d)
        csvs.append(pcsv)
        digs.append(pdig)
        os.chdir(root)
    return pl, csvs, digs, root


def high_pass_filter(signal, fc, time="none", plot=False):
    signal = np.asarray(signal)
    if str(type(time)) == "<class 'int'>":
        time = np.linspace(0, time, len(signal))
    elif str(type(time)) == "<class 'numpy.ndarray'>":
        pass
    signal = np.asarray(signal)
    w = fftfreq(signal.size, d=time[1] - time[0])
    f_signal = rfft(signal)
    cut_f_signal = f_signal.copy()
    cut_f_signal[(w < fc)] = 0
    cut_f_signal[(w < -fc)] = f_signal[(w < -fc)]
    cut_signal = irfft(cut_f_signal)
    if (plot == True):
        plt.subplot(221)
        plt.plot(time, signal)
        plt.subplot(222)
        plt.plot(w, f_signal)
        plt.subplot(223)
        plt.plot(time, cut_signal)
        plt.subplot(224)
        plt.plot(w, cut_f_signal)
        plt.show()
    return list(cut_signal)


def low_pass_filter(signal, fc, time="none", plot=False):
    signal = np.asarray(signal)
    if str(type(time)) == "<class 'int'>":
        time = np.linspace(0, time, len(signal))
    elif str(type(time)) == "<class 'numpy.ndarray'>":
        pass
    signal = np.asarray(signal)
    w = fftfreq(signal.size, d=time[1] - time[0])
    f_signal = rfft(signal)
    cut_f_signal = f_signal.copy()
    cut_f_signal[(w > fc)] = 0
    cut_f_signal[(w < (-1 * fc))] = 0
    cut_signal = irfft(cut_f_signal)
    if (plot == True):
        plt.subplot(221)
        plt.plot(time, signal)
        plt.subplot(222)
        plt.plot(w, f_signal)
        plt.subplot(223)
        plt.plot(time, cut_signal)
        plt.subplot(224)
        plt.plot(w, cut_f_signal)
        plt.show()
    return list(cut_signal)


def band_pass_filter(signal, f1, f2, time="none", plot=False):
    signal = np.asarray(signal)
    if str(type(time)) == "<class 'int'>":
        time = np.linspace(0, time, len(signal))
    elif str(type(time)) == "<class 'numpy.ndarray'>":
        pass
    signal = np.asarray(signal)
    w = fftfreq(signal.size, d=time[1] - time[0])
    f_signal = rfft(signal)
    cut_f_signal = f_signal.copy()
    cut_f_signal[(w < f1)] = 0
    cut_f_signal[(w > f2)] = 0
    cut_f_signal[(w < -f1)] = f_signal[(w < -f1)]
    cut_f_signal[(w < -f2)] = 0
    cut_signal = irfft(cut_f_signal)
    if (plot == True):
        plt.subplot(221)
        plt.plot(time, signal)
        plt.subplot(222)
        plt.plot(w, f_signal)
        plt.subplot(223)
        plt.plot(time, cut_signal)
        plt.subplot(224)
        plt.plot(w, cut_f_signal)
        plt.show()
    return list(cut_signal)


def dwt_filter(signal, time="none", wavelet="db1", plot=False):
    npsignal = np.asarray(signal)
    f_signal = rfft(npsignal)
    t_time = time
    if str(type(time)) == "<class 'int'>":
        time = np.linspace(0, time, len(signal))
    elif str(type(time)) == "<class 'numpy.ndarray'>":
        pass
    w = fftfreq(npsignal.size, d=time[1] - time[0])
    l, h = pywt.dwt(list(signal), wavelet=wavelet)
    if str(type(t_time)) == "<class 'int'>":
        time2 = np.linspace(0, t_time, len(l))
    elif str(type(t_time)) == "<class 'numpy.ndarray'>":
        pass
    f_l = rfft(l)
    f_h = rfft(l)
    w2 = fftfreq(np.asarray(l).size, d=time2[1] - time2[0])
    if (plot == True):
        plt.subplot(321)
        plt.plot(time, npsignal)
        plt.subplot(322)
        plt.plot(w, f_signal)
        plt.subplot(323)
        plt.plot(time2, l)
        plt.subplot(324)
        plt.plot(w2, f_l)
        plt.subplot(325)
        plt.plot(time2, h)
        plt.subplot(326)
        plt.plot(w2, f_h)
        plt.show()
    return l, h


def ps(signal, time, plot=False):
    signal = np.asarray(signal)
    time = np.linspace(0, time, len(signal))
    w = fftfreq(signal.size, d=time[1] - time[0])
    f_signal = rfft(signal)
    cut_f_signal = f_signal.copy()
    mx = max(cut_f_signal)
    mn = min(cut_f_signal)
    f_loc = 0
    if mx > (mn * -1):
        f_loc = list(cut_f_signal).index(mx)
    else:
        f_loc = list(cut_f_signal).index(mn)
    cut_f_signal[(w < (f_loc - 15))] = 0
    cut_f_signal[(w > (f_loc + 15))] = 0
    cut_signal = irfft(cut_f_signal)
    if (plot == True):
        plt.subplot(221)
        plt.plot(time, signal)
        plt.subplot(222)
        plt.plot(w, f_signal)
        plt.subplot(223)
        plt.plot(time, cut_signal)
        plt.subplot(224)
        plt.plot(w, cut_f_signal)
        plt.show()
    return cut_signal


def ecg_filter(ECG, plot=False):
    L = low_pass_filter(ECG, 900, 1)
    L, h = dwt_filter(L, 10, "db4")
    L, h = dwt_filter(L, 10, "db4")
    if plot:
        plt.plot(L)
        plt.show()
    return L


def ECG_prossecor(F_ECG):
    QRS_detector = F_ECG
    QRS_detector = high_pass_filter(QRS_detector, 100, 1)
    QRS_detector_hpf = QRS_detector
    sum = 0
    for s in QRS_detector:
        sum = sum + s
    av = sum / len(QRS_detector)
    for i, s in enumerate(QRS_detector):
        if QRS_detector[i] < av:
            QRS_detector[i] = av
    QRS_detector = band_pass_filter(QRS_detector, 1, 40, 1)
    QRS_detector = ps(QRS_detector, 1)
    sum = 0
    for s in QRS_detector:
        sum = sum + s
    av = sum / len(QRS_detector)
    R_locs = []
    c_peak = []
    for i, s in enumerate(QRS_detector):
        if i > 0:
            if s >= av:
                c_peak.append(i)

            elif s < av and QRS_detector[i - 1] > av and len(c_peak) > 0:
                c_qrsdp = []
                for s2 in c_peak:
                    c_qrsdp.append(QRS_detector_hpf[s2])
                R_locs.append(i - (len(c_qrsdp) - c_qrsdp.index(max(c_qrsdp))))
                c_peak = []
    R_locs_c = []
    unsafe = (5 * len(F_ECG)) / 100
    for l in R_locs:
        if l > unsafe and l < len(F_ECG) - unsafe:
            R_locs_c.append(l)
    R_locs = R_locs_c
    R_locs = R_locs[1:-1]

    rloc = []
    for i, s in enumerate(F_ECG):
        if i in R_locs:
            rloc.append(10)
        else:
            rloc.append(0)

    Q_locs = []
    for r in R_locs:
        cr = r
        while F_ECG[cr - 1] < F_ECG[cr]:
            cr = cr - 1
        cr = cr - 1
        Q_locs.append(cr)

    S_locs = []
    for r in R_locs:
        cr = r
        while F_ECG[cr + 1] < F_ECG[cr]:
            cr = cr + 1
        cr = cr + 1
        S_locs.append(cr)

    avg_pt = (R_locs[-1] - R_locs[0]) / (len(R_locs) - 1)
    p_p_d = round(avg_pt)
    pw = round(p_p_d / 3)
    P_locs = []
    for q in Q_locs:
        pws = q - pw
        pwa = []
        while pws < q:
            pwa.append(F_ECG[pws])
            pws = pws + 1
        P_locs.append(q - (len(pwa) - pwa.index(max(pwa))))

    PSE_detector = F_ECG

    PSE_detector = band_pass_filter(PSE_detector, 130, 160, 1)

    P_locs_sed = []
    for i, l in enumerate(P_locs):
        s = l - 5
        e = l + 5
        cw = PSE_detector[s:e]
        ctl = s + cw.index(max(cw)) + 1
        P_locs_sed.append(ctl)

    PS_locs = []
    for r in P_locs_sed:
        cr = r - 3
        while PSE_detector[cr - 1] < PSE_detector[cr]:
            cr = cr - 1
        PS_locs.append(cr)

    PE_locs = []
    for r in P_locs_sed:
        cr = r + 2
        while PSE_detector[cr + 1] < PSE_detector[cr]:
            cr = cr + 1
        PE_locs.append(cr)

    tw = round(p_p_d / 2)
    T_locs = []
    for s in S_locs:
        twe = s + tw
        tws = s
        twa = []
        while tws < twe:
            twa.append(F_ECG[tws])
            tws = tws + 1
        T_locs.append(s + twa.index(max(twa)))

    TSE_detector = F_ECG

    TSE_detector = band_pass_filter(TSE_detector, 70, 260, 1)

    T_locs_sed = []
    for i, l in enumerate(T_locs):
        s = l - 10
        e = l + 10
        cw = TSE_detector[s:e]
        ctl = s + cw.index(max(cw)) + 1
        T_locs_sed.append(ctl)

    TS_locs = []
    for r in T_locs_sed:
        cr = r - 5
        while TSE_detector[cr - 1] < TSE_detector[cr]:
            cr = cr - 1
        if S_locs[T_locs_sed.index(r)] >= cr:
            TS_locs.append(S_locs[T_locs_sed.index(r)] + 2)
        else:
            TS_locs.append(cr)

    TE_locs = []
    for r in T_locs_sed:
        cr = r + 5
        while TSE_detector[cr + 1] < TSE_detector[cr]:
            cr = cr + 4
        TE_locs.append(cr)
    return Q_locs, R_locs, S_locs, P_locs, PS_locs, PE_locs, T_locs, TS_locs, TE_locs


def plot_ecg(ECG, Q_locs, S_locs, PS_locs, PE_locs, TS_locs, TE_locs):
    det = []
    for i, s in enumerate(ECG):
        if i in Q_locs or i in S_locs or i in PS_locs or i in PE_locs or i in TS_locs or i in TE_locs:
            det.append(s)
        else:
            det.append(-1)
    f = figure(num=None, figsize=(100, 1), dpi=100, facecolor='w', edgecolor='k')
    plt.plot(ECG)
    plt.plot(det)
    move_figure(f, -10, 250)
    plt.show()


def ecg_fe(Q_locs, R_locs, S_locs, PS_locs, PE_locs, TS_locs, TE_locs, time):
    QRS_width = []
    for q, s in enumerate(S_locs):
        QRS_width.append(s - Q_locs[q])
    for i, s in enumerate(QRS_width):
        QRS_width[i] = QRS_width[i] * time / len(F_ECG)
    avg_qrs_w = 0
    for w in QRS_width:
        avg_qrs_w = avg_qrs_w + w
    avg_qrs_w = round(avg_qrs_w / len(QRS_width), 3)
    avg_pt = (R_locs[-1] - R_locs[0]) / (len(R_locs) - 1)
    avg_pt = avg_pt * time / len(F_ECG)
    HR = int(round(60 / avg_pt, 0))

    p_r_intervals = []
    for i, s in enumerate(PS_locs):
        p_r_intervals.append(round((Q_locs[i] - s) * time / len(F_ECG), 3))

    p_r_interval = 0
    for pri in p_r_intervals:
        p_r_interval = p_r_interval + pri
    p_r_interval = round(p_r_interval / len(p_r_intervals), 3)

    p_r_segments = []
    for i, s in enumerate(PE_locs):
        p_r_segments.append(round((Q_locs[i] - s) * time / len(F_ECG), 3))

    p_r_segment = 0
    for prs in p_r_segments:
        p_r_segment = p_r_segment + prs
    p_r_segment = round(p_r_segment / len(p_r_segments), 3)

    s_t_segments = []
    for i, s in enumerate(TS_locs):
        s_t_segments.append(round((s - S_locs[i]) * time / len(F_ECG), 3))

    s_t_segment = 0
    for sts in s_t_segments:
        s_t_segment = s_t_segment + sts
    s_t_segment = round(s_t_segment / len(s_t_segments), 3)

    s_t_intervals = []
    for i, s in enumerate(TE_locs):
        s_t_intervals.append(round((s - S_locs[i]) * time / len(F_ECG), 3))

    s_t_interval = 0
    for sti in s_t_intervals:
        s_t_interval = s_t_interval + sti
    s_t_interval = round(s_t_interval / len(s_t_intervals), 3)

    q_t_intervals = []
    for i, s in enumerate(Q_locs):
        q_t_intervals.append(round((TE_locs[i] - s) * time / len(F_ECG), 3))

    q_t_interval = 0
    for qti in q_t_intervals:
        q_t_interval = q_t_interval + qti
    q_t_interval = round(q_t_interval / len(q_t_intervals), 3)

    p_intervals = []
    for i, s in enumerate(PS_locs):
        p_intervals.append(round((PE_locs[i] - s) * time / len(F_ECG), 3))

    p_interval = 0
    for pi in p_intervals:
        p_interval = p_interval + pi
    p_interval = round(p_interval / len(p_intervals), 3)

    t_intervals = []
    for i, s in enumerate(TS_locs):
        t_intervals.append(round((TE_locs[i] - s) * time / len(F_ECG), 3))

    t_interval = 0
    for ti in t_intervals:
        t_interval = t_interval + ti
    t_interval = round(t_interval / len(t_intervals), 3)

    return avg_qrs_w, HR, p_r_interval, p_r_segment, s_t_segment, s_t_interval, q_t_interval, p_interval, t_interval, round(
        avg_pt, 3)



def ecg_enh(F_ECG,Q_locs,R_locs,S_locs,PS_locs,TS_locs,TE_locs,plot=False):
    max_r = F_ECG[Q_locs[0]]
    for s in R_locs:
        if F_ECG[s] > max_r:
            max_r = F_ECG[s]
    for s in R_locs:
        F_ECG[s] = max_r
    ct=10/len(F_ECG)
    F_ECG=F_ECG[(PS_locs[0]-(int(PS_locs[0]/5))):(TE_locs[-1]+(int((len(F_ECG)-TE_locs[-1])/5)))]
    if plot:
        plot_ecg(F_ECG,Q_locs,S_locs,PS_locs,PE_locs,TS_locs,TE_locs)
    ct=ct*len(F_ECG)
    return F_ECG,round(ct,3)


#################################################################################################################################

#################################################################################################################################
try:
    if not os.path.exists("dataset.csv"):
        df = pd.DataFrame.from_dict(
            {'QRS width': [], "HR": [], "P_R_interval": [], "P_R_segment": [], "S_T_interval": [],
             "S_T_segment": [], "Q_T_interval": [], "P_interval": [], "T_interval": [], "Diagnosis": []})
        df.to_csv('dataset.csv', mode='a', header=True, index=False)

    pl, csvs, digs, root = get_data()
    print(pl)
    pc = False
    if os.path.exists("log.txt"):
        pc = True
    for i, p in enumerate(pl):
        if pc:
            file = open("log.txt", "r")
            cp = file.readlines()[0][0:-1]
            if str(p) != str(cp):
                file.close()
                continue
            print("\n\n\n\n\n\n\n\n\n")
            file.close()
        print("\n***************************\n***************************")
        print(p)
        print("\n")
        for i2, csv in enumerate(csvs[i]):
            if pc:
                file = open("log.txt", "r")
                ccsv = file.readlines()[1]

                if ccsv != csv:
                    file.close()
                    continue
                file.close()
                pc = False
            file = open("log.txt", "w")
            file.write(str(p) + "\n" + str(csv))
            file.close()
            print("\n")
            print(csv)
            print("\n")
            data_set = pd.read_csv(root + "\\" + p + "\\" + csv)
            ECG_array = data_set.iloc[:, 0].values
            ECG_list = []
            for s in ECG_array:
                ECG_list.append(s)
            ECG = ECG_list
            F_ECG = ecg_filter(ECG)

            Q_locs, R_locs, S_locs, P_locs, PS_locs, PE_locs, T_locs, TS_locs, TE_locs = ECG_prossecor(F_ECG)
            e_ecg, mi = ecg_enh(F_ECG, Q_locs, R_locs, S_locs, PS_locs, TS_locs, TE_locs, True)

            avg_qrs_w, HR, p_r_interval, p_r_segment, s_t_segment, s_t_interval, q_t_interval, p_interval, t_interval, P_P = ecg_fe(
                Q_locs, R_locs, S_locs, PS_locs, PE_locs, TS_locs, TE_locs, 10)

            print("QRS_width:" + str(avg_qrs_w) + " S")
            print("HR:" + str(HR) + " BPM")
            print("P_R_interval:" + str(p_r_interval) + " S")
            print("P_R_segment:" + str(p_r_segment) + " S")
            print("S_T_interval:" + str(s_t_interval) + " S")
            print("S_T_segment:" + str(s_t_segment) + " S")
            print("Q_T_interval:" + str(q_t_interval) + " S")
            print("P_interval:" + str(p_interval) + " S")
            print("T_interval:" + str(t_interval) + " S")
            inp = input("Choose(y/n)=>")
            if inp == "n":
                continue
            df = pd.DataFrame.from_dict(
                {'QRS width': [avg_qrs_w], "HR": [HR], "P_R_interval": [p_r_interval], "P_R_segment": [p_r_segment],
                 "S_T_interval": [s_t_interval], "S_T_segment": [s_t_segment], "Q_T_interval": [q_t_interval],
                 "P_interval": [p_interval], "T_interval": [t_interval], "Diagnosis": [digs[i][i2]]})
            df.to_csv('dataset.csv', mode='a', header=False, index=False)
    print("[+]Operation completed successfuly.")
    input("Press Enter to exit")
    os.remove("log.txt")
except:
    print("[-]Something went wrong.")
    input("Press Enter to exit")