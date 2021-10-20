import os
import copy

from os import link
import urllib.request
from urllib.request import urlopen
from bs4 import BeautifulSoup
from selenium import webdriver
import time

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
tfe = tf.contrib.eager
tfe.enable_eager_execution() 

import numpy as np
from tensor2tensor import problems
from tensor2tensor import models
from tensor2tensor import problems
from tensor2tensor.utils import trainer_lib
from tensor2tensor.data_generators import text_encoder

from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import registry

import textwrap

import sys
sys.path.append("nmt-en-tr")
import nmt_en_tr

from app import app
from tables import Results
from db_config import mysql
from flask import flash, render_template, request, redirect

#Machine Translation section

model_path = 'en2tr'

data_dir = os.path.join(model_path, 'data')
ckpt_dir = os.path.join(model_path, 'model')

en2tr_problem = problems.problem("translate_en_tr")
encoders = en2tr_problem.feature_encoders(data_dir)

ckpt_path = tf.train.latest_checkpoint(ckpt_dir)

def encode(input_str, output_str=None):
  """Input str to features dict, ready for inference"""
  inputs = encoders["inputs"].encode(input_str) + [1]  # add EOS id 
  batch_inputs = tf.reshape(inputs, [1, -1, 1])  # Make it 3D.
  return {"inputs": batch_inputs}

def decode(integers):
  """List of ints to str"""
  integers = list(np.squeeze(integers))
  if 1 in integers:
    integers = integers[:integers.index(1)]
  if len(integers) == 1:
    return encoders["inputs"].decode(integers)

  return encoders["inputs"].decode(np.squeeze(integers))

model_name = "transformer"
hparams_set = "transformer_tpu"

Modes = tf.estimator.ModeKeys

hparams = trainer_lib.create_hparams(hparams_set, data_dir=data_dir, problem_name="translate_en_tr")

translate_model = registry.model(model_name)(hparams, Modes.EVAL)

def translate(inputs, beam_size = 5, alpha = 0.6, **kwargs):
  encoded_inputs = encode(inputs)
  with tfe.restore_variables_on_create(ckpt_path):
    model_output = translate_model.infer(encoded_inputs, **kwargs)["outputs"]
  if len(model_output.shape) == 2:
    return decode(model_output)
  else:
    return [decode(x) for x in model_output[0]]
  
def translate_and_display(input):
  output = translate(input)
  print('\n  '.join(textwrap.wrap("Input: {}".format(input), 80)))
  print()
  print('\n  '.join(textwrap.wrap("Output: {}".format(output), 80)))

# Backend section below

@app.route('/translate')
def add_user_view():
	return render_template('add.html')

@app.route('/edit_user')
def edit_user_view():
	return render_template('edit.html')
		
@app.route('/add', methods=['POST'])
def add_user():
	try:		
		_link = request.form['inputName']
		# validate the received values
		if _link and request.method == 'POST':
			print(_link)

			html = urlopen(_link).read()
			soup = BeautifulSoup(html, features="html.parser")

			for script in soup(["script", "style"]):
				script.extract()
			
			text = soup.get_text()

			lines = (line.strip() for line in text.splitlines())
			chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
			text = '\n'.join(chunk for chunk in chunks if chunk)
			texts = text.split("\n")

			mystr = html.decode("utf8")

			translationTable = str.maketrans("ğĞıİöÖüÜşŞçÇ", "gGiIoOuUsScC")

			line_s = mystr.split("\n")
			
			for i in texts:
				if i.isdecimal():
					continue

				length = len(i)
				
				for line in line_s:
					index = line_s.index(line)

					if line == line_s[-1]:
						new_line = "aaaaa"
						break
					elif index == 0:
						old_line = "aaaaa"
					else:
						old_line = line_s[index-1]
						new_line = line_s[index+1]

					start_index = line.find(i)

					if start_index < 0:
						continue
					else:
						if ">" in line[start_index-4:start_index] and "<" in line[start_index+length:start_index+length+4]:
							line_s[index] = line[:start_index] + translate(i).translate(translationTable) + line[start_index+length:]
							break
						elif old_line.replace(" ", "")[-4:].find(">") >= 0 and new_line.replace(" ", "")[:4].find("<") >= 0:
							line_s[index] = line[:start_index] + translate(i).translate(translationTable) + line[start_index+length:]
							break
						elif old_line.replace(" ", "")[-4:].find(">") >= 0 and "<" in line[start_index+length:start_index+length+4]:
							line_s[index] = line[:start_index] + translate(i).translate(translationTable) + line[start_index+length:]
							break
						elif ">" in line[start_index-4:start_index] and new_line.replace(" ", "")[:4].find("<") >= 0:
							line_s[index] = line[:start_index] + translate(i).translate(translationTable) + line[start_index+length:]
							break

			mystr = "\n".join(line_s)

			file = open("translated.html","w")
			file.write(mystr)

			flash("Translated file is ready.")

			return redirect('/translate')
		else:
			return 'Error while adding user'
	except Exception as e:
		print(e)

		
@app.route('/')
def users():
	try:
		return render_template('users.html')
	except Exception as e:
		print(e)

if __name__ == "__main__":
    app.run()
