# Salt_and_Paper

Задание: изображение размером M x N с шумом "Salt and Papper" реализовать медианный фильтр с использованием текстурной памяти.

Изображение размером 128 x 35 

![test(128x85)_blured](https://github.com/vasser2323/HPC_Labs/assets/73202398/4b6eaf3d-261e-48e4-bdf2-03814106a509)

Изображение размером 256 x 171

![test(256x171)_blured](https://github.com/vasser2323/HPC_Labs/assets/73202398/352ffd98-158b-45e0-9c03-0ebc670c2a8f)

Изображение размером 512 x 341

![test(512x341)_blured](https://github.com/vasser2323/HPC_Labs/assets/73202398/4c7d2648-bac4-4590-8c3f-783a5d612bcc)

Изображение размером 1024 x 683 

![test(1024x683)_blured](https://github.com/vasser2323/HPC_Labs/assets/73202398/110daec8-d8e9-4f12-bb82-079260af682d)

График сравнения скоростей

![image](https://github.com/vasser2323/HPC_Labs/assets/73202398/9ddc3bb1-5c8e-4d8f-967a-bd183f8f144b)

График ускорения

![image](https://github.com/vasser2323/HPC_Labs/assets/73202398/0311faf8-6603-448a-aa42-97daf2b0984e)

Вывод: 

Реализовал median filter на CPU и GPU, провел анализ для 4 изображений с разным разрешением, 
подсчитал время выполнения и ускорения, заметно, что ускрение так же растет, при увеличении разрешения картинки, 
что объясняет сложность обработки больших массивов для cpu
