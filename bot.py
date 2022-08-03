import discord
from discord.utils import get
from discord.ext import commands
import sklearn
import numpy as np
import pandas as pd
from keras_preprocessing import image
import tensorflow as tf

from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as pyplot
dict1 = {"1": "Jacinto de los bosques", "2": "Ranúnculo bulboso", "3": "fárfara", "4": "Primula", "5": "Crocus", "6": "Narciso", "7": "Margarita", "8": "Diente de leon", "9": "Tablero de damas", "10": "Iris", "11": "Lirio de los valles", "12": "Viola tricolor", "13": "campanilla de invierno", "14": "Girasol", "15": "Flor del lazo atigrado", "16": "Tulipan", "17": "Anémonas"}
#spanish dictonary

intents = discord.Intents.default()
intents.members = True  
bot = commands.Bot(command_prefix='/', intents=intents)
model = tf.keras.models.load_model('weights-improvement-97-0.96')

@bot.event
async def on_ready():

    print("uff")
    await bot.change_presence(activity=discord.Game(name="Ver flores"))

@bot.event
async def on_message(message):
    if message.content.startswith('!flor'):
        #get attachment
        attachment = message.attachments[0]
        #get image
        await attachment.save('image.png')
        images= image.load_img("image.png", target_size=(224, 224))
        images = image.img_to_array(images)
        images = np.expand_dims(images, axis=0)
        a = model.predict(images)

        a = a.tolist()
        print(a)
        v = 0
        h = 0
        m = []
        for i in a[0]:
            v += 1
            print("{}: {}".format(v, i))
            m.append(i)

        s = np.asarray(m).argmax()
        w = np.asarray(m).max()
        await message.channel.send(f"Esta flor es: {dict1[str(s+1)]}")
        """
        for o in m:
            h += 1
            if o > 0:
                if o/w * 100 >30 and o/w * 100 != 100:
                    await message.channel.send(dict1[str(h-1)] + f" {o/w * 100}%")"""




        #await message.channel.send(f"Esta flor es: {dict1[str(s)]}")
    if message.content == "Cual es esta flor?":
        c_channel = discord.utils.get(message.guild.text_channels, name='flor')
        messages = await c_channel.history(limit=2).flatten()
        att = messages[1].attachments[0]
        await att.save('image.png')
        images= image.load_img("image.png", target_size=(224, 224))
        images = image.img_to_array(images)
        images = np.expand_dims(images, axis=0)
        a = model.predict(images)

        a = a.tolist()
        print(a)
        v = 0
        h = 0
        m = []
        for i in a[0]:
            v += 1
            print("{}: {}".format(v, i))
            m.append(i)

        s = np.asarray(m).argmax()
        w = np.asarray(m).max()
        await message.channel.send(f"Esta flor es: {dict1[str(s+1)]}")
        """
        for o in m:
            h += 1
            if o > 0:
                if o/w * 100 >30 and o/w * 100 != 100:
                    await message.channel.send(dict1[str(h-1)] + f" {o/w * 100}%")"""




        #await message.channel.send(f"Esta flor es: {dict1[str(s)]}")




bot.run("TOKEN")
