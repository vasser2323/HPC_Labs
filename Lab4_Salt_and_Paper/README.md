# Salt_and_Paper

Задание: изображение размером M x N с шумом "Salt and Papper" реализовать медианный фильтр с использованием текстурной памяти.

Изображение размером 128 x 35 

![test(128x85)_blured](https://github.com/vasser2323/Salt_and_Paper/assets/73202398/da993b12-5f91-4ddf-ba17-5344ec6553a0)

Изображение размером 256 x 171

![test(256x171)_blured](https://github.com/vasser2323/Salt_and_Paper/assets/73202398/10b3a639-42cb-4fdc-bd5f-6614f41c891f)

Изображение размером 512 x 341

![test(512x341)_blured](https://github.com/vasser2323/Salt_and_Paper/assets/73202398/851b5234-66d4-4467-9137-3f6be5df0c22)

Изображение размером 1024 x 683 

![test(1024x683)_blured](https://github.com/vasser2323/Salt_and_Paper/assets/73202398/fa8e9184-ce20-4d2a-9206-bf5354b8ef59)

График сравнения скоростей

![image](https://github.com/vasser2323/Salt_and_Paper/assets/73202398/d462d289-da3a-4558-8cc0-f0ca58489218)

График ускорения

![image](https://github.com/vasser2323/Salt_and_Paper/assets/73202398/aaf9b1e4-5af6-477f-92b7-d27033b2452b)


Вывод: 

Реализовал median filter на CPU и GPU, провел анализ для 4 изображений с разным разрешением, 
подсчитал время выполнения и ускорения, заметно, что ускрение так же растет, при увеличении разрешения картинки, 
что объясняет сложность обработки больших массивов для cpu
