Overview
========

In the following, several more extensive data analysis pathways involving EEGLAB functions are sketched. The first pathway
illustrates the use of ASR through a convenience wrapper function (clean_asr) that simplifies some of the logistics
of running the core functions. The second pathway shows the use of an all-in-one cleaning function that, besides ASR
also performs some basic pre- and post-processing such as removing bad channels and time windows (clean_artifacts). 
The last example invokes a function that can be used to visually compare raw versus cleaned data (vis_artifacts).

All of these examples accept data in the form of EEGLAB dataset structs (see http://sccn.ucsd.edu/wiki/A05:_Data_Structures). 
The easiest way to supply the necessary data is to use the EEGLAB toolbox (freely available from http://sccn.ucsd.edu/eeglab/).
The data set used in these examples is the freely available Sternberg data set #1. This data set can be downloaded
from headit.org by navigating to the study named "Modified Sternberg Working Memory Task", going to subject #1,
and then downloading the first recording of the subject, here called  "eeg_recording_1.SMA" -- the following link
points directly to this data set but may be deprecated by the database maintainers at some point: 
http://headit-beta.ucsd.edu/attachments/e9e23582-a236-11e2-9420-0050563f2612/download).

These examples use additional code (in the extras directory) developed at the Swartz Center for which we cannot provide the 
same level of support as for the core ASR functions.

Using ASR through the convenience wrapper
=========================================

addpath(pwd);        % add current directory to the path
addpath extras;      % add extras to the path
eeglab;              % start EEGLAB if not done so already (freely available from sccn.ucsd.edu/eeglab/)

raw = pop_biosig('eeg_recording_1.SMA');        % load a sample data set (freely available from: headit.org or http://headit-beta.ucsd.edu/recording_sessions/e67fc0e4-a236-11e2-9420-0050563f2612)
highpassed = clean_drifts(raw,[0.25 0.75]);		% high-pass filter it using a 0.25-0.75 Hz transition band
cleaned = clean_asr(highpassed,2);			    % run the ASR algorithm using a threshold of 2 standard deviations (somewhat less conservative than default; see docs for more options)


Using ASR as part of the all-in-one-cleaning
============================================

addpath(pwd);        % add current directory to the path
addpath extras;      % add extras to the path
eeglab;              % start EEGLAB if not done so already (freely available from sccn.ucsd.edu/eeglab/)

raw = pop_biosig('eeg_recording_1.SMA');   % load a sample data set (freely available from: headit.org or http://headit-beta.ucsd.edu/recording_sessions/e67fc0e4-a236-11e2-9420-0050563f2612)
cleaned = clean_artifacts(raw,'BurstCriterion',2,'WindowCriterion','off'); % clean the data with final removal of windows disabled and same threshold as before


Visually comparing the raw and cleaned data
===========================================

vis_artifacts(cleaned,raw);       % this function has important keyboard shortcuts to toggle the display -- see documentation

