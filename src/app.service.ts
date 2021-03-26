import { Injectable, OnModuleInit } from '@nestjs/common';
import * as use from '@tensorflow-models/universal-sentence-encoder';
import * as tf from '@tensorflow/tfjs-node';
import { join } from 'path';

@Injectable()
export class AppService implements OnModuleInit {
  onModuleInit() {
    console.log('run');
    this.test();
  }
  getHello(): string {
    return 'Hello World!';
  }

  public async testLoad(): Promise<void> {
    const encoder = await use.load();
    const model = await tf.loadLayersModel(
      'file://' + join(__dirname, '../storage', 'model.json'),
    );

    for (const text of this.testData) {
      const xPredict = await this.encodeData(encoder, [{ text }]);
      const prediction = await (model.predict(xPredict) as tf.Tensor).data();

      console.log(prediction);
      // if (prediction[0] > threshold) {
      //   return 'BOOK';
      // } else if (prediction[1] > threshold) {
      //   return 'RUN';
      // } else {
      //   return null;
      // }
    }
  }

  public async test(): Promise<void> {
    const encoder = await use.load();
    const xTrain = await this.encodeData(encoder, this.trainData);
    const yTrain = tf.tensor2d(
      this.trainData.map((t) => [
        t.intent === 'buy' ? 1 : 0,
        t.intent === 'none' ? 1 : 0,
      ]),
    );

    const model = tf.sequential();
    model.add(
      tf.layers.dense({
        inputShape: [xTrain.shape[1]],
        activation: 'softmax',
        units: 2,
      }),
    );

    model.compile({
      loss: 'categoricalCrossentropy',
      optimizer: tf.train.adam(0.001),
      metrics: ['accuracy'],
    });

    await model.fit(xTrain, yTrain, {
      epochs: 200,
    });

    for (const text of this.testData) {
      const xPredict = await this.encodeData(encoder, [{ text }]);
      const prediction = await (model.predict(xPredict) as tf.Tensor).data();

      console.log(prediction);
    }

    model.save('file://' + join(__dirname, '../storage'));
  }

  private async encodeData(encoder, datasets): Promise<any> {
    const sentences = datasets.map((t) => t.text.toLowerCase());
    const embeddings = await encoder.embed(sentences);
    return embeddings;
  }

  private testData = [
    'where can I buy your dress?',
    'how much was that dress?',
    'You look so beautiful in that dress',
    'I love you, you are gorgeous',
    'I’ve been looking into this game recently and it looks pretty cool. I want a game with fun combat and a decent amount of customization, and it seems like Dead Cells has that. I’ve never played a rogue like before, but I have played Hollow Knight, which I didn’t really enjoy because you always feel lost and deaths are really frustrating, especially when it takes 10 minutes getting back to your shade and you feel just as weak as you did before. I would buy again if I needed to.',
  ];

  private trainData = [
    {
      text: 'buy',
      intent: 'buy',
    },
    {
      text: 'buying',
      intent: 'buy',
    },
    {
      text: 'purchase',
      intent: 'buy',
    },
    {
      text: 'buy that',
      intent: 'buy',
    },
    {
      text: 'buy this',
      intent: 'buy',
    },
    {
      text: 'buy it',
      intent: 'buy',
    },
    {
      text: 'where to buy',
      intent: 'buy',
    },
    {
      text: 'where can I buy that',
      intent: 'buy',
    },
    {
      text: 'how much',
      intent: 'buy',
    },
    {
      text: 'Woah yess❤️❤️',
      intent: 'none',
    },
    {
      text: 'Happy birthday to the most beautiful woman on Instagram ! 💗💗',
      intent: 'none',
    },
    {
      text: 'Your body 😍',
      intent: 'none',
    },
    {
      text: 'Angel',
      intent: 'none',
    },
    {
      text:
        'I saw this swimsuit on a website and wasn’t convinced, but you rock it and now I want to buy it',
      intent: 'buy',
    },
    {
      text: 'So cute..sexy, sexy',
      intent: 'none',
    },
    {
      text: 'Ooo i need to look like u girl!! Gonna hit the gym now',
      intent: 'none',
    },
    {
      text: 'Beautiful👍💜',
      intent: 'none',
    },
    {
      text: '😱😱💖💖',
      intent: 'none',
    },
    {
      text:
        'What is your workout routine? It would be lovely if you could share 😍😍😍',
      intent: 'none',
    },
  ];
}
