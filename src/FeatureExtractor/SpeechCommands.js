// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/*
A class that extract features from SpeechCommands
*/

import * as tf from '@tensorflow/tfjs';
import * as tfjsSpeechCommands from '@tensorflow-models/speech-commands';

import { saveBlob } from '../utils/io';
import callCallback from '../utils/callcallback';
import { getTopKClassesFromArray } from '../utils/gettopkclasses';

class SpeechCommands {
  constructor(callback) {
    this.speechCommandsModel = null;
    /**
     * Boolean value to check if the model is predicting.
     * @public
     * @type {boolean}
     */
    this.isPredicting = false;
    /**
     * String that specifies how is the Extractor being used. 
     *    Possible values are 'regressor' and 'classifier'
     * @type {String}
     * @public
     */
    this.usageType = null;
    this.ready = callCallback(this.loadModel(), callback);
    this.options = {};
    this.model = null;
  }

  async loadModel() {
    this.speechCommandsModel = tfjsSpeechCommands.create('BROWSER_FFT');
    await this.speechCommandsModel.ensureModelLoaded();
    return this;
  }

  classification(options) {
    this.usageType = 'classifier';
    this.model = this.speechCommandsModel.createTransfer('transfer');
    if (typeof options === 'object') {      
      Object.assign(this.options, options);
    }
    return this;
  }

  /**
   * Adds a new sound example to SpeechCommands
   * @param {String || function} labelOrCallback 
   * @param {function} cb 
   */
  async addExample(labelOrCallback, cb) {
    let label;
    let callback = cb;

    if (typeof labelOrCallback === 'string' || typeof labelOrCallback === 'number') {
      label = labelOrCallback;
    } else if (typeof labelOrCallback === 'function') {
      callback = labelOrCallback;
    }

    return callCallback(this.addExampleInternal(label), callback);
  }

  async addExampleInternal(label) {
    if (this.model.isListening()) this.model.stopListening();
    await this.model.collectExample(label);
    return this;
  }

  /**
   * Retrain the model with the provided images and labels using the 
   *    models original features as starting point.
   * @param {function} onProgress  - A function to be called to follow 
   *    the progress of the training.
   */
  async train(onProgress) {
    this.examplesCount = this.model.countExamples();
    if (!this.examplesCount || Object.keys(this.examplesCount).length <= 0) {
      throw new Error('Add some examples before training!');
    }
    this.isPredicting = false;

    if (this.usageType === 'classifier') {
      this.wordLabels = this.model.wordLabels();
      await this.model.train({
        epochs: 25,
        callback: {
          onEpochEnd: async (epoch, logs) => {
            console.log(`Epoch ${epoch}: loss=${logs.loss}, accuracy=${logs.acc}`);
            onProgress(logs.loss.toFixed(5));
          },
          onTrainEnd: () => onProgress(null),
        }
      });
    }
  }

  /**
   * Classifies an sond example based on a new retrained model. 
   *    .classification() needs to be used with this.
   * @param {function} cb 
   */
  async classify(numOrCallback = null, cb) {
    if (this.usageType !== 'classifier') {
      throw new Error('SpeechCommands Feature Extraction has not been set to be a classifier.');
    }

    let numberOfClasses = this.wordLabels.length;
    let callback;

    if (typeof numOrCallback === 'number') {
      numberOfClasses = numOrCallback;
    } else if (typeof numOrCallback === 'function') {
      callback = numOrCallback;
    }

    if (typeof cb === 'function') {
      callback = cb;
    }
    return this.classifyInternal(numberOfClasses, callback);
  }

  classifyInternal(topk, cb) {
    if (this.model.isListening()) this.model.stopListening();
    return this.model.listen(result => {
      if (result.scores) {
        const classes = getTopKClassesFromArray(result.scores, topk, this.wordLabels)
          .map(c => ({ label: c.className, confidence: c.probability }));
        return cb(null, classes);
      }
      return cb(`ERROR: Cannot find scores in result: ${result}`);
    }, this.options)
    .catch(err => {
      return cb(`ERROR: ${err.message}`);
    });
  }

  async load(filesOrPath = null, callback) {
    if (typeof filesOrPath === 'string') {
      this.model = tfjsSpeechCommands.create('BROWSER_FFT', undefined, `${filesOrPath}/model.json`, `${filesOrPath}/metadata.json`);
      await this.model.ensureModelLoaded();
      this.wordLabels = this.model.wordLabels();
      if (callback) {
        callback();
      }
    }
    return this.model;
  }

  async save(callback) {
    if (!this.model) {
      throw new Error('No model found.');
    }

    this.model.save(tf.io.withSaveHandler(async (data) => {
      console.log('data: ', data)
      const meta = this.model.getMetadata();
      console.log('meta: ', meta);
      await saveBlob(JSON.stringify(data), 'model.json', 'text/plain');
      await saveBlob(JSON.stringify(meta), 'metadata.json', 'text/plain');

      if (callback) {
        callback();
      }
    }));
  }
}

export default SpeechCommands;
