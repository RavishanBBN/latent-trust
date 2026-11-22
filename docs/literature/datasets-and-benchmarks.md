# Datasets, benchmarks and FDD baselines for D5 (verified 2026-11-14)

## Benchmarks with control inputs
| Item | Key facts | Controls / faults | Link | Notes |
|---|---|---|---|---|
| **Quadruple tank** (Johansson 2000, IEEE TCST 8(3):456–465, doi 10.1109/87.845876) | **Minimum-phase setting P−:** A₁=A₃=28, A₂=A₄=32 cm²; a₁=a₃=0.071, a₂=a₄=0.057 cm²; k=(3.33, 3.35) cm³/(V·s); γ=(0.70, 0.60); h°=(12.4, 12.7, 1.8, 1.4) cm at 3.00 V. **Non-minimum-phase setting P+:** k=(3.14, 3.29); γ=(0.43, 0.34); h°=(12.6, 13.0, 4.8, 4.9) cm at 3.15 V | 2 pump voltages; faults injected in simulation | https://scilit.com/publications/6d560f56730f3274f5eed53ecbb199db | Parameters confirmed via two secondary sources that cite the paper (arXiv 2304.04763 Tables 1–2; Saeed et al. 2010, Strathprints 42140). Our simulator reaches (12.26, 12.78, 1.63, 1.41) cm |
| **Tennessee Eastman** (Downs & Vogel 1993, Comput. Chem. Eng. 17(3):245–255, doi 10.1016/0098-1354(93)80018-I) | Simulator has 12 XMV, 41 XMEAS, 15 IDV | Yes | Code: https://depts.washington.edu/control/LARRY/TE/download.html | |
| Revised TEP (Bathelt, Ricker, Jelali 2015, IFAC-PapersOnLine 48(8):309–314, doi 10.1016/j.ifacol.2015.08.199) | Solver-independent random states; adds XMEAS 42–73 and IDV 21–28 | Yes | https://folk.ntnu.no/skoge/prost/proceedings/adchem2015/media/papers/0010.pdf | Code `temexd_mod.zip` in Ricker's archive |
| **Rieth et al. 2017** TEP data (Harvard Dataverse, doi 10.7910/DVN/6C3JR1) | 500 runs per fault, faults 1–20 plus fault-free; **52 variables = 41 XMEAS + 11 XMV**; train runs 25 h, test runs 48 h, sampled every 3 min | Yes (11 XMV) | https://doi.org/10.7910/DVN/6C3JR1 | Fault-onset time not verified. The common practice is to drop faults 3, 9 and 15 |
| **Reinartz et al. 2021** extended TEP (Comput. Chem. Eng. 149:107281, doi 10.1016/j.compchemeng.2021.107281) | 28 faults, 6 operating modes, 500 seeds per fault, 100 h runs, setpoint changes and mode transitions | Faults ✅; XMV not confirmed | https://data.dtu.dk/articles/dataset/Tennessee_Eastman_Reference_Data_for_Fault-Detection_and_Decision_Support_Systems/13385936 | Check XMVs and the licence |
| **PRONTO** (Stief et al., J. Process Control 79:41–55, 2019, doi 10.1016/j.jprocont.2019.04.009) | **Real** Cranfield multiphase-flow rig; 1 Hz process data plus alarms, ultrasonic, pressure and video | Valve positions logged; air and water setpoints varied (20–200 Sm³/h; 0.1–3.5 kg/s); faults: air leak, air blockage, diverted flow, slugging | https://zenodo.org/records/1341583 (**CC BY 4.0, free**) | **Best real dataset found.** Confirm that setpoints are separate channels |
| Cranfield rig, earlier dataset (Ruiz-Cárcel et al. 2015, Control Eng. Practice 42:74–88) | Real rig with seeded faults; 2016 companion paper confirms flow setpoints and valve positions at 1 Hz | Yes | IEEE DataPort (paywalled) | Use PRONTO instead |
| LBNL HVAC faults (Granderson et al., Sci. Data 10, 2023, doi 10.1038/s41597-023-02197-w) | ~257 fault cases, 1-min data; AHU, RTU, FCU, VAV, plant | Control signals present (FCU valve confirmed) | https://faultdetection.lbl.gov/ | **Mostly simulated** |
| SWaT (Goh et al., CRITIS 2016, LNCS 10242:88–99) | 51 attributes (24 sensors, 27 actuators), 1 Hz, 11 days | Actuator states ✅; labels are **cyber-attacks** | iTrust request form | No redistribution |
| SKAB | — | **No control inputs**, so unsuitable | https://github.com/waico/SKAB | |

## Fault-detection baselines and reviews
| Key | Citation | Link |
|---|---|---|
| qin2003 | Qin (2003). Statistical process monitoring: basics and beyond. *J. Chemometrics* 17(8–9):480–502 | https://doi.org/10.1002/cem.800 |
| yin2012 | Yin, Ding, Haghani, Hao, Zhang (2012). A comparison study of basic data-driven fault diagnosis and process monitoring methods on the benchmark TEP. *J. Process Control* 22(9):1567–1581 | https://doi.org/10.1016/j.jprocont.2012.06.009 |
| chiang2001 | Chiang, Russell, Braatz (2001). *Fault Detection and Diagnosis in Industrial Systems*. Springer | https://doi.org/10.1007/978-1-4471-0347-9 |
| lomov2021 | Lomov, Lyubimov, Makarov, Zhukov (2021). Fault detection in TEP with temporal deep learning models. *J. Ind. Inf. Integr.* 23:100216 | https://doi.org/10.1016/j.jii.2021.100216 |
