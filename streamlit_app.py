## -------------------------------
## ====    Import smthng      ====
## -------------------------------

##Libraries for Streamlit
##--------------------------------
import streamlit as st
import io
from scipy.io import wavfile as scipy_wav
from PIL import Image

##Libraries for prediction
##--------------------------------
import numpy as np
import matplotlib.pyplot as plt
from itertools import count

import tensorflow as tf
from tensorflow.keras import models


## For debug
##--------------------------------
import os 
import psutil
# import gc
# import sys
# gc.collect()
# sys.modules[__name__].__dict__.clear()





id_logo = Image.open("TypoMeshDarkFullat3x.png")
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.image(id_logo)





##-----------------------------------
## Load 2 models
##-----------------------------------

# @st.cache
# def load_model_vgg_s8_64_it4BrK13JvI2():
#     return tf.keras.models.load_model("./tf_models/modelTN_vgg_s8-64_l2233_s09e5_data_it4BrK13JvI2")
# @st.cache
# def load_model_vgg_s8_64_it4BrK13JvI2_uChunks():
#     return tf.keras.models.load_model("./tf_models/model_uChunks_04sec_TN_vgg_s8-64_l2233_s09e5_data_it4BrK13JvI2/")
# reloaded_model_vgg_s8_64_it4BrK13JvI2 = load_model_vgg_s8_64_it4BrK13JvI2()
# reloaded_model_vgg_s8_64_it4BrK13JvI2_uChunks = load_model_vgg_s8_64_it4BrK13JvI2_uChunks()


tf.keras.backend.clear_session() # clean memory, remove models

reloaded_model_vgg_s8_64_it4BrK13JvI2_SingleSpectr = tf.keras.models.load_model("./tf_models/modelTN_vgg_s8-64_l2233_s09e5_data_it4BrK13JvI2")
reloaded_model_vgg_s8_64_it4BrK13JvI2_uChunks = tf.keras.models.load_model("./tf_models/model_uChunks_04sec_TN_vgg_s8-64_l2233_s09e5_data_it4BrK13JvI2/")





st.markdown("<h1 style='text-align: center; color: grey;'>Select Audio Sample for Analysis</h1>", 
            unsafe_allow_html=True)
## -------------------------------
## ====  Select and load data ====
## -------------------------------


st.subheader("Select one of the samples")

selected_provided_file = st.selectbox(label="", 
                            options=["example of a cutting event", "example of a background sound"]
                            )


st.subheader("or Upload an audio file in WAV format")
st.write("if a file is uploaded, previously selected samples are not taken into account")

uploaded_audio_file = st.file_uploader(label="Select a short WAV file < 5 sec", 
                                        type="wav", 
                                        accept_multiple_files=False, 
                                        key=None, 
                                        help=None, 
                                        on_change=None, 
                                        args=None, 
                                        kwargs=None, 
                                        disabled=False)


uploaded_audio_file_4debug = uploaded_audio_file #this expression must be before Data Switch



## Data Switch is here
##--------------------------------
if uploaded_audio_file is not None:
    # st.write("YEP")
    ##for single spectrogram model
    SSModel_audio_arr_sr, SSModel_audio_arr = scipy_wav.read(uploaded_audio_file)
    ##for uChunk model
    uCModel_audio_data_to_predict = list(scipy_wav.read(uploaded_audio_file))
    uCModel_filepath_to_predict = uploaded_audio_file.name

else:
    # st.write("NOPE") #444debug
    if selected_provided_file == "example of a cutting event":
        example_of_a_cutting_event_file_path = 'Event_Bibox_Piezo_Session3-105.wav'
        SSModel_audio_arr_sr, SSModel_audio_arr = scipy_wav.read(example_of_a_cutting_event_file_path)
        uCModel_audio_data_to_predict = list(scipy_wav.read(example_of_a_cutting_event_file_path))
        uCModel_filepath_to_predict = example_of_a_cutting_event_file_path
    if selected_provided_file == "example of a background sound":
        example_of_a_background_sound_file_path = 'Background_BPeizo_Talk_BKG-51.wav' 
        SSModel_audio_arr_sr, SSModel_audio_arr = scipy_wav.read(example_of_a_background_sound_file_path)
        uCModel_audio_data_to_predict = list(scipy_wav.read(example_of_a_background_sound_file_path))
        uCModel_filepath_to_predict = example_of_a_background_sound_file_path




st.markdown("<h1 style='text-align: center; color: grey;'>===================================</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: grey;'>Analysis with Single Spectrogram ML Model</h1>", unsafe_allow_html=True)

## If stereo, then do averaging over channels 
##-------------------------------------------
if len(SSModel_audio_arr.shape) > 1:
    SSModel_audio_arr = np.mean(SSModel_audio_arr, axis=1, dtype=int)

## Normalize values of audio 
##--------------------------
SSModel_audio_arr = (SSModel_audio_arr - np.mean(SSModel_audio_arr))/ np.std(SSModel_audio_arr)


## Convert to virtula file to play it 
##-------------------------------------------
virtualfile = io.BytesIO()
scipy_wav.write(virtualfile, rate=SSModel_audio_arr_sr, data=SSModel_audio_arr)
uploaded_audio_file = virtualfile





## -------------------------------
## ====   Show selected data  ====
## -------------------------------

st.subheader("Show the data selected for analysis")
st.header(" ")
st.header(" ")
st.markdown("<h2 style='text-align: center; color: grey;'>Show the data selected for analysis</h2>", 
            unsafe_allow_html=True)

# st.write("Listen the loaded data")
st.markdown(" ##### _Listen the loaded data_")
st.audio(uploaded_audio_file, format='audio/wav')
st.markdown(" ##### _Waveform of the loaded data_")
fig_wf_28, ax_wf_28 = plt.subplots(1,1, figsize=(5, 2))
ax_wf_28.plot(SSModel_audio_arr)
ax_wf_28.grid('True')
st.pyplot(fig_wf_28)





# ----------------------------------------
# ==== Functions to make spectrograms ====
# ----------------------------------------

def get_Single_spectrogram( waveform, sampling_rate ):
    waveform_1d = tf.squeeze(waveform)
    waveform_1d_shape = tf.shape(waveform_1d)
    n_samples  = waveform_1d_shape[0]
    spectrogram = tf.signal.stft(
                        tf.squeeze(tf.cast(waveform, tf.float32)),
                        frame_length=tf.cast(n_samples/100, dtype=tf.int32),
                        frame_step=tf.cast(n_samples/100/4, dtype=tf.int32),
                        )
    spectrogram = tf.abs(spectrogram)
    l2m = tf.signal.linear_to_mel_weight_matrix(
                        num_mel_bins=125,
                        num_spectrogram_bins=tf.shape(spectrogram)[1],
                        sample_rate=sampling_rate,
                        lower_edge_hertz=0,
                        upper_edge_hertz=22000,
                        )
    spectrogram = tf.matmul(spectrogram, l2m)
    spectrogram = tf.math.divide(spectrogram, tf.math.reduce_max(spectrogram) )
    spectrogram = tf.math.add(spectrogram, tf.math.reduce_min(spectrogram) )
    spectrogram = tf.math.add(spectrogram, 0.01 )
    spectrogram = tf.math.log(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=-1)
    spectrogram = tf.transpose(spectrogram, perm=(1,0,2))
    spectrogram = spectrogram[::-1, :, :]
    return spectrogram


#default values which is used to plot spectrogram and as input_shape unless it is changed according to a model specs
#once again! this value might be overwritten for a specific model
single_spectrogram_shape_to_analyze = (64*2*1, 64*4*1)


def single_spectrogram_resize(spectrogram):
    return tf.image.resize(spectrogram, single_spectrogram_shape_to_analyze)



# ----------------------------------------
# ==== Create Dataset of spectrograms ====
# ----------------------------------------


single_spectrogram_arr = get_Single_spectrogram(SSModel_audio_arr, SSModel_audio_arr_sr)
single_spectrogram_arr_resized = single_spectrogram_resize(single_spectrogram_arr)


## Show spectrogram
##--------------------------------
st.markdown(" ##### _Spectrogram of the  loaded data_")
fig_sp_28, ax_sp_28 = plt.subplots(1,1, figsize=(5, 2))
ax_sp_28.imshow(single_spectrogram_arr_resized)
st.pyplot(fig_sp_28)




## -------------------------------
## ====    Apply ML model     ====
## -------------------------------

st.header(" ")
st.header(" ")
st.markdown("<h2 style='text-align: center; color: grey;'>Analysis with ML model</h2>", 
            unsafe_allow_html=True)




# -----------------------------------
# ==== Predict with loaded Model ====
# -----------------------------------
# multiple if-statements are needed to adjust inputs for models

ssm_y_pred_2 = reloaded_model_vgg_s8_64_it4BrK13JvI2_SingleSpectr.predict(np.expand_dims(single_spectrogram_arr_resized, 0))
ssm_audio_data_predicted_label = np.round(ssm_y_pred_2[0][0], decimals=2)
ssm_ypred4debug = ssm_y_pred_2



 

st.subheader("Prediction:")

ssm_pred_index = np.round(ssm_audio_data_predicted_label, decimals=0).astype(int)
ssm_results_options = ['No cutting sound detected.',
                    'Canvas cutting is DETECTED.']

st.markdown(f"### _{ssm_results_options[ssm_pred_index]}_")
# st.write(f"Result: {ssm_results_options[ssm_pred_index]}")



## Output of more tech data
##--------------------------------
st.markdown(f"#### Model Prediction Output: {ssm_audio_data_predicted_label}")
print(f"Model Prediction Output: {ssm_audio_data_predicted_label}")

st.markdown(f"Raw Model Prediction Output: {ssm_ypred4debug}")
print(f"Raw Model Prediction Output: {ssm_ypred4debug}")

st.markdown(f"#### Model Prediction Output Index: {ssm_pred_index}")
print(f"Model Prediction Output Index: {ssm_pred_index}")

st.write(uploaded_audio_file_4debug)
print(uploaded_audio_file_4debug)





































## /////////////////////////////////////////////////////////////////////////////////
## /////////////////////////////////////////////////////////////////////////////////
## /////////////////////////////////////////////////////////////////////////////////
## /////////////////////////////////////////////////////////////////////////////////
## /////////////////////////////////////////////////////////////////////////////////
## /////////////////////////////////////////////////////////////////////////////////


st.markdown("<h1 style='text-align: center; color: grey;'>===================================</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: grey;'>Analysis with uChunks ML Model</h1>", unsafe_allow_html=True)


## FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
## FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
## ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
## FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
## FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF


## -------------------------------------------
## ==== Function to split audio to chunks ====
## -------------------------------------------

# uChunk_len_sec = 0.4 #desired chunk length in sec #0.4is default
uChunk_len_sec = st.slider('Set uChunk length [sec]. Model trained for 0.4 sec.', min_value=0.1, max_value=1.5, value=0.4, step=0.1)
uChunk_acceptance_factor = 0.8 #if a uChunk has length of at least 80% of the desired length, we accept it

def split_audio_to_uChunks(audio_data):
    list_for_dataset=[]
    
    # for cc_ae in audio_data_list:
        # cc_ae[0] - tuple (sampling_rate, audio_np_array)
        #### cc_ae[1] - label <===NOT in prediction script
        #### cc_ae[2] - file_path <===NOT in prediction script
        # cc_ae[1] - file_path
        
        
    #>> Get audio <<
    arr_to_split = audio_data[0][1] #cc_ae[0][1]
    sr_of_arr = audio_data[0][0] #cc_ae[0][0]      
    
    #>> If stereo, average the channels <<
    if len(arr_to_split.shape) > 1:
        arr_to_split = np.mean(arr_to_split, axis=1, dtype=int)
    
    #>> Split audio to uChunks <<
    uChunk_len_audiosamples = uChunk_len_sec * sr_of_arr
    split_indices = np.arange(start=0, stop=arr_to_split.shape[0]-1, step=uChunk_len_audiosamples, dtype='int')
    uChunks_list = np.split(arr_to_split, split_indices[1:], axis=0)
    uChunks_list_lenghts = list(map(len, uChunks_list))

    #>> Check whether last chunk size is ok <<
    if uChunks_list_lenghts[-1] < uChunk_acceptance_factor*uChunk_len_audiosamples:
        uChunks_list = uChunks_list[:-1]
        uChunks_list_lenghts = list(map(len, uChunks_list))

    #>> Make list of indices of chunks <<
    uChunk_subindices = np.arange(start=0, stop=len(split_indices), step=1)
        
    #>> Normalize each audio uChunk <<
    uChunks_list = [cc.astype('float32') for cc in uChunks_list]
    uChunks_list = [(cc-cc.mean())/cc.std() for cc in uChunks_list]
    
    #>> Extend the list by adding sampling rate info <<
    # uChunks_list_ext = list(zip(uChunks_list, [sr_of_arr]*len(uChunks_list), [cc_ae[1]]*len(uChunks_list) ))
        # Copy bcoz it is a little different for prediction: the last element is file_path but not label
    uChunks_list_ext = list(zip(uChunks_list, 
                                [sr_of_arr]*len(uChunks_list), 
                                [audio_data[1]]*len(uChunks_list), 
                                uChunk_subindices ))
    

    #>> Add the new list to the common list dedicated to dataset <<
    if len(uChunks_list_ext)>0:
        list_for_dataset.extend(uChunks_list_ext)

    return list_for_dataset
        

## ----------------------------------------
## ==== Functions to make spectrograms ====
## ----------------------------------------

@tf.function
def get_uChunk_spectrogram( waveform, sampling_rate, file_path, uChunk_index ):
    waveform_1d_shape = tf.shape(waveform)
    n_samples  = waveform_1d_shape[0]
    spectrogram = tf.signal.stft(
                        tf.squeeze(waveform),
                        frame_length=tf.cast(n_samples/100, dtype=tf.int32),
                        frame_step=tf.cast(n_samples/100/4, dtype=tf.int32),
                        )
    spectrogram = tf.abs(spectrogram)
    l2m = tf.signal.linear_to_mel_weight_matrix(
                        # num_mel_bins=125,
                        num_mel_bins=40,
                        num_spectrogram_bins=tf.shape(spectrogram)[1],
                        sample_rate=sampling_rate,
                        lower_edge_hertz=0,
                        upper_edge_hertz=22000,
                        )
    spectrogram = tf.matmul(spectrogram, l2m)
    spectrogram = tf.math.divide(spectrogram, tf.math.reduce_max(spectrogram) )
    spectrogram = tf.math.add(spectrogram, tf.math.reduce_min(spectrogram) )
    spectrogram = tf.math.add(spectrogram, 0.01 )
    spectrogram = tf.math.log(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=-1)
    spectrogram = tf.transpose(spectrogram, perm=(1,0,2))
    spectrogram = spectrogram[::-1, :, :]
    return spectrogram, file_path, uChunk_index


uChunk_spectrogram_shape_to_analyze = (64*2*1, 64*1*1)

@tf.function
def uChunk_spectrogram_resize(spectrogram, file_path, uChunk_index):
    return tf.image.resize(spectrogram, uChunk_spectrogram_shape_to_analyze), file_path, uChunk_index




## DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD
## AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
## ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
## TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
## AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

audio_and_filepath_data_list_to_predict = [uCModel_audio_data_to_predict, uCModel_filepath_to_predict]
st.code(audio_and_filepath_data_list_to_predict)

virtualfile = io.BytesIO()
scipy_wav.write(virtualfile, rate=uCModel_audio_data_to_predict[0], data=uCModel_audio_data_to_predict[1])
st.audio(uploaded_audio_file, format='audio/wav')

fig_wf, ax_wf = plt.subplots(1,1, figsize=(5, 2))
ax_wf.plot(uCModel_audio_data_to_predict[1])
ax_wf.grid('True')
st.pyplot(fig_wf)



## -------------------------------------------------
## ==== Apply Function to split audio to chunks ====
## -------------------------------------------------

list_for_dataset_to_predict = split_audio_to_uChunks(audio_and_filepath_data_list_to_predict)

# st.code( len(list_for_dataset_to_predict) )
# st.code(list_for_dataset_to_predict)


## ---------------------------------------
## ==== Make tf.dataset from py list ====
## ---------------------------------------
dataset_to_predict_p0_arr = tf.data.Dataset.from_generator(lambda: [cc[0] for cc in list_for_dataset_to_predict], tf.float32, output_shapes=[None])
dataset_to_predict_p2_sr = tf.data.Dataset.from_tensor_slices([cc[1] for cc in list_for_dataset_to_predict])
dataset_to_predict_p3_filepath = tf.data.Dataset.from_tensor_slices([cc[2] for cc in list_for_dataset_to_predict])
dataset_to_predict_p4_subindices = tf.data.Dataset.from_tensor_slices([cc[3] for cc in list_for_dataset_to_predict])

dataset_to_predict_original = tf.data.Dataset.zip((dataset_to_predict_p0_arr, 
                                                   dataset_to_predict_p2_sr, 
                                                   dataset_to_predict_p3_filepath, 
                                                   dataset_to_predict_p4_subindices))

# st.code(list(dataset_to_predict_original.take(1)))


## ---------------------------------------------------------
## ==== Apply Functions to make spectrograms on dataset ====
## ---------------------------------------------------------
dataset_to_predict = dataset_to_predict_original.map(get_uChunk_spectrogram)
dataset_to_predict = dataset_to_predict.map(uChunk_spectrogram_resize)


## ------------------------------------------------
## ==== See samples of the uChunk spectrograms ====
## ------------------------------------------------
fig_sp, ax_sp = plt.subplots(1, len(list_for_dataset_to_predict), figsize=(15, 12))
# for spectrogram, path_filename, uChunk_index in dataset_to_predict.take(2):
ax_counter = 0
for spectrogram, path_filename, uChunk_index in dataset_to_predict:
    # st.code(f"{path_filename.numpy().decode('utf-8').split('/')[-1]} : {uChunk_index}")
    ax_sp[ax_counter].imshow(spectrogram)
    ax_counter += 1
st.pyplot(fig_sp)


##-P-R-P-R-P-R-P-R-P-R-P-R-P-R-P-R-P-R-P-R-P-R-P-R-P-R-P-R-P-R-P-R-P-R-P-R-P-R-P-R-P-R-P-R-P-R-P-R-P
##E-D-E-D-E-D-E-D-E-D-E-D-E-D-E-D-E-D-E-D-E-D-E-D-E-D-E-D-E-D-E-D-E-D-E-D-E-D-E-D-E-D-E-D-E-D-E-D-E- 
##-I-C-I-C-I-C-I-C-I-C-I-C-I-C-I-C-I-C-I-C-I-C-I-C-I-C-I-C-I-C-I-C-I-C-I-C-I-C-I-C-I-C-I-C-I-C-I-C-I
##T-I-T-I-T-I-T-I-T-I-T-I-T-I-T-I-T-I-T-I-T-I-T-I-T-I-T-I-T-I-T-I-T-I-T-I-T-I-T-I-T-I-T-I-T-I-T-I-T-
##-O-N-O-N-O-N-O-N-O-N-O-N-O-N-O-N-O-N-O-N-O-N-O-N-O-N-O-N-O-N-O-N-O-N-O-N-O-N-O-N-O-N-O-N-O-N-O-N-O

st.subheader("Prediction:")

## -------------------------------------------------------
## ==== Do predictions for each uChunk in the dataset ====
## -------------------------------------------------------

for cc, cc_path, cc_index in dataset_to_predict.take(1): dataset_test_element_shape = cc.shape.as_list()
dataset_test_element_shape = np.append(0, dataset_test_element_shape)

test_audio = np.empty(dataset_test_element_shape)
test_file_name = []
test_uChunk_index = []

for audio, file_path, uChunk_index in dataset_to_predict:
    test_audio = np.append(test_audio, np.expand_dims(audio.numpy(), 0), axis=0) #used here, but to nexessary
    test_file_name.append(file_path.numpy().decode("utf-8").split('/')[-1])
    test_uChunk_index.append(uChunk_index.numpy())

y_pred = reloaded_model_vgg_s8_64_it4BrK13JvI2_uChunks.predict(test_audio)
y_pred_index = y_pred.squeeze().round(0).astype(int)

data_classes = ['00_Backgrounds_', 
                '01_Events______']

y_pred_classes = [data_classes[i] for i in y_pred_index]

predictions_for_files = [["%.3f" % cc3.item(), data_classes[cc2], сс1, cc4] for сс1, cc2, cc3, cc4 in zip(test_file_name, y_pred_index, y_pred, test_uChunk_index)]

# print(f"preiction: {*predictions_for_files}")
# print(*predictions_for_files, sep='\n')
st.markdown(f"### _Prediction for each chunk:_")
st.markdown(f"##### [Probability], [Predcited Class], [File Name], [uChunk Index]")

for cc_pr in predictions_for_files:
    st.code(cc_pr)


n_of_samples = y_pred_index.shape[0]
# print(f'\nN_of samples: {n_of_samples}')

n_of_events = sum(y_pred_index == np.ones(y_pred_index.shape))
test_accuracy_events = n_of_events / len(y_pred_index)
st.markdown(f'Amount of samples per class\n Background class: {n_of_samples-n_of_events} ({1-test_accuracy_events:.0%})\n Event class: {n_of_events} ({test_accuracy_events:.0%})\n')

# print(f'Model raw predictions:\n {y_pred}\n')



## ----------------------------------------------------------------------------
## ==== Aggregate predictions of uChunks for each Audio_Sample (each file) ====
## ----------------------------------------------------------------------------
## Logic:
## ------
## Scan the sequence of uChunk prediction for an Audio_Sample with a defined window length.
## The window length is defined in 'desired_nof_window_samples'. The default value is 3.
## The windows are overlapping, bcoz the step is 1.
## The predictions within a window are summed (np.sum is applied, basically averaging).
## If any of the values is more than a threshold (0.5 multiplied by number of elements used in window) 
##      we say that the Audio_Sample consist of a Cutting Event (does not matter in which part).
##
## Probability reduction:
## ----------------------
## If number of uChunks is less than the window length, the prediction is done anyway, 
## but the probability is rediced accourding to a formula.
## The reduction logic is following.
## In case of the number of uChunks equal or more to the window length, there is no reduction.
## In case of the number of uChunks is 1, the prediction of the uChunk must be at least 90% (or 95% depending on setting),
## to consider it as a Cutting Event.
## For all intermediate number of uChunks the reduction factor is scalled accordingly.

from itertools import count

for file_cc in list(set(test_file_name)):
    ## >> Get indices of one file <<
    occurance_indices = [cci for cci, ccj in zip(count(), test_file_name) if ccj == file_cc]

    ## >> Filter the data according to indices <<
    test_file_name_ccf = [test_file_name[cc] for cc in occurance_indices]
    y_pred_index_ccf = [y_pred_index[cc] for cc in occurance_indices]
    y_pred_ccf = [np.round(y_pred[cc].item(), 3) for cc in occurance_indices]
    test_uChunk_index_ccf = [test_uChunk_index[cc] for cc in occurance_indices]
    
    st.write("...................................")
    st.markdown(f"filename: {set(test_file_name_ccf).pop()}")
    # print(f"y_pred_index_ccf: {y_pred_index_ccf}")
    # print(f"y_pred_ccf: {y_pred_ccf}")
    # print(f"test_uChunk_index_ccf: {test_uChunk_index_ccf}")
    
    ## >> Sort or Make sure that the uChunks are in correct sequence <<
    # sorted_sequence_y_pred = [(cc1,cc2) for cc1, cc2 in sorted(zip(test_uChunk_index_ccf, y_pred_ccf), reverse=False)] #keep to check data later
    sorted_sequence_y_pred = [cc2 for cc1, cc2 in sorted(zip(test_uChunk_index_ccf, y_pred_ccf), reverse=False)]

    ## >> Calculate area (integral step) for each uChunk pair <<
    # sequence_integral_y_pred  = np.diff(sorted_sequence_y_pred)/2 + sorted_sequence_y_pred[:-1]
    ## good idea but we have too little uChunks/samples to calculate intergral/area values
    
    ## >> Calculate integral values for each window of uChunk predictions array <<
    # desired_nof_window_samples = 3
    desired_nof_window_samples = st.slider('Set number of uChunks for averaging. 3 is default.', min_value=2, max_value=7, value=3, step=1)
    actual_nof_window_samples = min(desired_nof_window_samples, len(sorted_sequence_y_pred)) #3(desired_nof_window_samples) uChunks are taken into account for prediction
    summ_window_threshold = 0.5 * actual_nof_window_samples #0.5 is a threshold for a single value prediction, like for usual output
    array_of_summ_window = np.array([])

    for ci in range(len(sorted_sequence_y_pred) - actual_nof_window_samples + 1): #+1 bcoz range() excludes the last value
        ci_data_in_window = sorted_sequence_y_pred[ci:ci+actual_nof_window_samples]
        array_of_summ_window = np.append(array_of_summ_window, np.sum(ci_data_in_window))

    ## >> Reduction of probability <<
    ## This is needed for the case when an audio sample is so short that there are not enough uChunks
    ## Predicted probability is linearly reduced if the number of uChunks is less than the desired.
    ## For example: desired n_of_uchunks = 3. 
    ##     If we get 3 uChunks at minimum, then the probability will not be changed.
    ##     If we get 1 uChunk at maximum, then the probability will be reduced in proportion from 0.9 to 0.5.
    ##          This means in case of one uChunk the probability must be at least 90% to classify it positive.
    ##     If we get intermediate values (between 1 and desired value), the reduction factor will be scaled linearly.
    ## formulae: 
    ##        delta = (p_from - p_to ) * (desired_nof_window_samples - actual_nof_window_samples) / (desired_nof_window_samples - 1)
    ##        k = (p_from - delta) / p_from
    ##        p_reduced = p * k
    p_from = 0.95   #tuning parameter: some reference value of probability, which should be reduced to p_to value
    p_to = 0.50      #tuning parameter: some reference value of probability, to which should be reduced p_from value
    reduction_delta = (p_from - p_to ) * (desired_nof_window_samples - actual_nof_window_samples) / (desired_nof_window_samples - 1)
    k_reduction_slope = (p_from - reduction_delta) / p_from
    array_of_summ_window_reduced = array_of_summ_window * k_reduction_slope

    ## >> Calculate final of probability of prediction <<
    ## If any of the values within the sound sample is classified positive, then the whole sample is positive
    total_y_prediction = any(np.array(array_of_summ_window_reduced) > summ_window_threshold)
    st.markdown(f"### Does Audio Sample consist of Event: **{total_y_prediction}**")
    st.markdown(f"The value is _True_ if average probability of ANY {desired_nof_window_samples} chunks in sequence is more then 0.5.")    
    st.markdown(f"Take into account that for the number of chunks less than {desired_nof_window_samples} reduction factor is applied (see the code).")

## /////////////////////////////////////////////////////////////////////////////////
## /////////////////////////////////////////////////////////////////////////////////
## /////////////////////////////////////////////////////////////////////////////////
## /////////////////////////////////////////////////////////////////////////////////
## /////////////////////////////////////////////////////////////////////////////////
## /////////////////////////////////////////////////////////////////////////////////














st.markdown("<h3 style='text-align: center; color: grey;'>===================================</h1>", unsafe_allow_html=True)

## Show memory usage
##--------------------------------
process = psutil.Process(os.getpid())
mem_used_by_app = process.memory_info().rss /1024 /1024  # in MBytes 
print(f"The app uses: {mem_used_by_app} MB")
st.markdown(f"#### The app uses: {mem_used_by_app} MB")



