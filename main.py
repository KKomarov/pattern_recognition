import tkinter as tk
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def load_dataset():
    import sklearn.datasets
    mnist = sklearn.datasets.fetch_mldata('MNIST original', data_home='./data')
    return mnist


def save_as_image(arr, fn='file.png'):
    arr2 = arr.copy()
    arr2.resize((28, 28))
    Image.fromarray(arr2).save(fn)


class NeuralNet:
    def __init__(self):
        self.dbn = MLPClassifier(
            # [28 * 28, 300, 10],
            (700,),
            learning_rate="constant",
            # learn_rate_decays=0.9,
            max_iter=20,
            activation='logistic',
            verbose=1,
        )

        self.nn_file = 'data/nn.pkl'

    @lru_cache()
    def split_dataset(self):
        mnist = load_dataset()
        return train_test_split(mnist.data / 255.0, mnist.target, train_size=30000, test_size=1000)

    def train(self):
        X_train, X_test, y_train, y_test = self.split_dataset()
        self.dbn.fit(X_train, y_train)

    def test(self):
        X_train, X_test, y_train, y_test = self.split_dataset()
        y_pred = self.dbn.predict(X_test)
        p = accuracy_score(y_test, y_pred)
        print('accuracy:', p)

    def save_nn(self):
        joblib.dump(self.dbn, self.nn_file, compress=9)

    @lru_cache()
    def load_nn(self):
        try:
            self.dbn = joblib.load(self.nn_file)
            self.test()
        except:
            print('failed to load net.')


class MyGui:
    def __init__(self, nn):
        self.nn = nn
        root = tk.Tk()
        root.title("Neural network course work")
        frame = tk.Frame(root)
        frame.pack()
        self.w, self.h = 300, 50
        self.canvas = tk.Canvas(frame, bg='black', width=self.w, height=self.h)
        self.canvas.pack()
        self.label = tk.Label(frame, fg="dark green", height=1, width=40)
        self.label.pack()
        button = tk.Button(frame, text="Test", command=self.on_button)
        button.pack(side=tk.LEFT)
        tk.Button(frame, text="Clear", command=self.clear_canvas).pack(side=tk.LEFT)
        tk.Button(frame, text="Train", command=self.train_nn).pack(side=tk.LEFT)
        self.img = Image.new(mode='L', size=(self.w, self.h), color=0)
        self.draw = ImageDraw.Draw(self.img)
        self.coord = None
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset)
        self.root = root

    def train_nn(self):
        self.nn.train()
        self.nn.test()
        self.nn.save_nn()

    def clear_canvas(self):
        self.canvas.delete('all')
        self.draw.rectangle((0, 0) + (self.w, self.h), fill=0)

    def on_button(self):
        # self.img.save('from_canvas.png')
        self.nn.load_nn()
        arr = self.as_array()
        # print(arr)
        # p = self.nn.dbn.predict(arr)
        p = parallel(lambda a: self.nn.dbn.predict(a), arr, 1)
        print(p)
        self.label.config(text=str(p))

    def reset(self, event):
        self.coord = None

    def as_array(self):
        seg = Segmenter(self.w, self.h)
        return seg.process(self.img)

    def paint(self, event):
        line_width = 3
        paint_color = 'white'
        new_coord = (event.x, event.y)
        if self.coord:
            self.draw.line(self.coord + new_coord, fill=255, width=line_width)
            self.canvas.create_line(*self.coord, *new_coord,
                                    width=line_width, fill=paint_color,
                                    capstyle=tk.ROUND, smooth=tk.TRUE, splinesteps=36)
        self.coord = new_coord


def parallel(f, arr, n):
    from multiprocessing.pool import ThreadPool
    pool = ThreadPool(processes=n)
    m = min(len(arr), n)
    if m <= 0:
        return []
    ts = []
    chunk_size = len(arr) // m
    for i in range(0, len(arr), chunk_size):
        t = pool.apply_async(f, (arr[i:i + chunk_size],))
        ts.append(t)
    r = []
    for t in ts:
        r.extend(t.get())
    return r


def as_array(img):
    return np.fromstring(img.tobytes(), dtype=np.uint8) / 255.


class Segmenter:
    def __init__(self, w, h):
        self.w, self.h = w, h

    def process(self, img):
        start = None
        symbol_ranges = []
        for x in range(self.w):
            s = all(0 == img.getpixel((x, y)) for y in range(self.h))
            if s and start is not None:
                symbol_ranges.append((start, x))
                start = None
            if not s and start is None:
                start = x
        if start is not None:
            symbol_ranges.append((start, self.w))

        arrays = []
        for start, end in symbol_ranges:
            s = img.crop((start, 0) + (end, self.h))
            maxw = max(s.size)
            s1 = Image.new(mode='L', size=(maxw, maxw), color=0)
            s1.paste(s, box=((maxw - s.size[0]) // 2, (maxw - s.size[1]) // 2))
            s2 = s1.resize((28, 28))
            arrays.append(as_array(s2))
            s2.save("%s.png" % len(arrays))
        return arrays


def dslice(arr, fr, to, step, width):
    return [arr[i:i + width] for i in range(fr, to, step)]


if __name__ == '__main__':
    # save_as_image(data.data[5900])
    nn = NeuralNet()
    gui = MyGui(nn)
    gui.root.mainloop()
