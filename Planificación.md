# Sesión 1: 19 de Mayo

## Qué es una red neuronal artificial y por qué ~~chucha~~ necesito saber vectores y matrices para aprenderlas

Ejemplos de problemas a resolver:
- Diagnóstico de enfermedades
- Clasificación de imágenes
- Clasificación de texto

Arquitectura de una red neuronal. Mencionar los ejemplos con énfasis en el hecho de que los datos deben ser tratados como vectores que se transforman bajo parámetros.

## Herramientas para esto

- Instalación de Julia
- Ejemplo de cálculo vectorial de promedios

## Tarea 1

Usted tiene un curso de 25 estudiantes con dos tareas y dos pruebas, el promedio de las tareas pondera el $40\%$ de la nota del curso (NC), con el promedio de las pruebas aportando al otro $60\%$. Al final del semestre se han rendido todas las evaluaciones, las notas hasta el momento son las siguientes:
```
P1 = [1.0, 3.6, 2.3, 2.5, 3.3, 4.2, 5.7, 1.0, 5.3, 3.0, 6.2, 5.2, 3.3, 6.1, 6.2, 1.4, 2.4, 3.4, 2.5, 4.3, 6.4, 5.0, 1.6, 2.4, 6.6]
P2 = [2.1, 4.1, 6.4, 4.1, 4.0, 5.7, 5.3, 3.6, 5.1, 1.0, 4.0, 5.0, 3.8, 3.3, 2.7, 1.0, 3.2, 3.9, 2.6, 1.6, 2.5, 2.4, 2.6, 1.0, 6.3]
T1 = [2.5, 5.3, 6.0, 5.3, 6.0, 6.0, 5.5, 6.0, 5.3, 3.1, 5.3, 4.2, 4.0, 6.0, 4.4, 6.0, 6.1, 6.0, 4.2, 3.4, 6.3, 6.0, 4.0, 4.4, 6.4]
T2 = [4.8, 5.3, 4.7, 2.4, 4.4, 4.8, 7.0, 4.4, 6.4, 3.9, 4.7, 3.3, 4.3, 2.8, 4.2, 2.4, 2.0, 3.3, 3.8, 3.5, 2.6, 4.8, 4.5, 3.3, 2.5]
```
1. Calcule con esto un vector que contenga la nota (NC) cada estudiante del curso y grafíquelo en un histograma. Para esto, si ```NC``` es el nombre del vector que contiene las notas, le basta con correr las siguientes líneas de código:
```
using Plots
histogram(NC)
```
Para un gráfico un poco más presentable, puede utilizar este código en lugar del otro
```
using Plots
histogram(NC, bins=range(1, 7, length=7), color=:orange, label="NC", title="Distribución de Notas del Curso", xlabel="Notas", ylabel="Cantidad", background=:gray15)
```

2. Al parecer no le fue muy bien al curso, por lo que se ha decidido hacer un examen recuperativo (ER) que ponderará el $30\%$ con la nota (NC) ponderando el otro $70\%$. Es decir, si (NF) es la nota final, esta se calcula con la fórmula:
$$(NF) = 0.7(NC) + 0.3(ER)$$
Calcule un vector que contenga la menor nota que cada estudiante necesitaría obtener en el examen (ER) para aprobar el curso (es decir, $(NF) = 3.95$).

**Extra:** Para más información sobre la presentación de histogramas en Julia, ver el [siguiente link](https://docs.juliaplots.org/latest/series_types/histogram/).


# Sesión 2: 26 de Mayo

**Recapitulación:**

- Recordar la arquitectura de una red neuronal.
- ¿Cómo sé que los parámetros que tengo son los correctos?
- Función de error. Cómo evalúo lo malos que son mis parámetros.
- ¿Cómo obtengo los buenos parámetros? -> Optimización (y no de pocas variables)

## Una introducción a la optimización

- Partimos con una variable ¿Cómo podemos encontrar el mínimo?
- Regla de Fermat. Si el punto es mínimo, una hormiga parada en él cree que la función es plana. -> Derivada.

$$\overline{x} \text{ es mínimo local} \implies f'(\overline{x}) = 0$$

Así que necesitamos derivadas.

## ¿Y qué son las derivadas?

Dibujo de la recta, definición con límite. Ejemplo de la derivada de $x^2$, derivada de $x^3$. Intuición de por qué están bien. Fórmula para $(x^n)'$.

Mencionar el principio de que el signo de la derivada determina el crecimiento de la función -> Se puede optimizar mediante método de descenso.

**Nota:** En la sesión siguiente vamos a implementar el método del descenso en una variable y señalar cómo se extiende a varias variables.

## Tarea 2

1. Calcular por definición la derivada de la función
$$f(x) = \frac1x$$

2. Teniendo el resultado de la pregunta anterior y asumiendo como conocida la regla de derivación de la suma $(f + g)' = f' + g'$, encuentre aquel valor $\overline{x} > 0$ tal que la derivada de la función
$$h(x) = \frac1x + x$$
se anula en $\overline{x}$.

3. Deduzca de las dos preguntas anteriores que para todo $x > 0$, se cumple la siguiente desigualdad
$$x + \frac1x \geq 2$$

# Sesión 3: 2 de Junio

## Notaciones

Sobre las dimensiones:

- $d \in \N$: Dimensionalidad de los datos (256 pixeles)
- $n \in \N$: Dimensión de llegada (4 letras)
- $p \in \N$: Cantidad de parámetros (todos los pesos)
- $N \in \N$: Número de datos

Sobre las colecciones de datos:

- $\{x_i\}_{i=1}^N$: Datos de prueba ($x_i \in \R^d$)
- $\{y_i\}_{i=1}^N$: Etiquetas ($y_i \in \R^n$)
- $W = \{w_i\}_{i=1}^p$: Colección de parámetros ($w_i \in \R$)

Sobre las funciones:

- Dado $W \in \R^p$, tenemos un $f_W: \R^d \to \R^n$ red neuronal que posee los parámetros de $W$.
- Cada $W \in \R^p$ posee un error asociado $e(W)$ que viene en general dado por
$$e(W) = \sum_{i=1}^N d(f_W(x_i), y_i)$$
En que $d$ es una función que mide distancia entre vectores de $\R^n$. Así, queda definida una $e: \R^p \to \R^+$ que buscamos minimizar.

# Sesión 4: 23 de Junio

Partimos desde el mensaje de que _hay que aprender optimización_ y llegamos a la idea de una derivada. Calculamos e interpretamos el ejemplo de $f(x) = x^2$. Aquí es donde se dio la tarea 2.

# Sesión 5: 30 de Junio

## Tarea 3

En esta tarea vamos a obtener toda la información que podamos sobre la función
$$g(x) = x^3 - 3x$$
Se le sugiere no buscar su gráfico sino hasta haber completado la tercera parte. Esto le permitirá tener una noción de lo que se puede y no se puede deducir hasta ahora con las herramientas que tenemos.

1. Calcular por definición la derivada de la función $f(x) = x^3$. Para esto, le será útil recordar que
$$(a+b)^3 = a^3 + 3a^2b + 3ab^2 + b^3$$

2. Ahora, vuelva a calcular la derivada de la misma función $f(x) = x^3$, pero esta vez escribiéndola como un producto $f(x) = x \cdot x^2$ y utilizando la regla del producto vista en la sesión:
$$(f \cdot g)' = f' \cdot g + f \cdot g'$$

3. Utilizando el resultado obtenido y las reglas de derivación vistas, obtenga la derivada de la función $g(x) = x^3 - 3x$. Obtenga los $x$ donde $g'(x)$ se anula y analice también sus signos. Deduzca con esta información los máximos y mínimos (locales) de la función $g$. Esboce un gráfico con esta información.

# Sesión 6: 21 de Julio

Profundizando sobre el concepto de la derivada, se discutieron los criterios de la primers y segunda derivada.

# Sesión 7: 25 de Julio

Se discutió terminología de funciones regulares y se resumieron las reglas de derivación (salvo cadena), de demostró la regla del producto.

# Sesión 8: 14 de Agosto

## Ejemplos

Se resolvieron los siguientes problemas

- Calcular la derivada de $f(x) = 3x^2 + 2x + 1$
- Calcular la derivada de $f(x) = x + \frac{1}{x}$ y utilizar esto para probar la desigualdad
$$x + \frac{1}{x} \geq 2$$

# Sesión 9: 18 de Agosto

Se cierran las ideas anteriormente discutidas sobre criterios de primera y segunda derivada con un ejemplo, que consistió en graficar la función

$$f(x) = x^3 - 3x$$

## Tarea 4

En esta tarea, vamos a esbozar el gráfico de la función $f: \mathbb{R} \to \mathbb{R}$ dada por

$$
f(x) = \frac{1}{x^2 + 1}
$$

Puede utilizar como referencia los pasos seguidos en la sesión 9, los cuales se detallan a continuación:

1. **(Puntos de corte y signos)** Obtenga los puntos de intersección de la función $f$ con el eje $Y$ ¿Existe un punto se intersección con el eje $X$? Luego, analice los signos de la función mediante una tabla de signos.

    *Pista:* ¿Se puede factorizar la expresión del denominador?

2. **(Crecimiento, máximos y mínimos)** Obtenga la derivada de la función $f$ utilizando la regla del cuociente. Analice los signos de esa expresión (mediante tabla) para obtener los intervalos en que $f'$ es positiva y negativa. Con esto, deduzca los intervalos de crecimiento de $f$ y a partir de ello, sus valores máximos y mínimos en caso de que existan.

    *Para pensar:* Piense en el crecimiento (o decrecimiento) de la función para $x$ grande ¿Esto tiene sentido considerando el signo de $f$ obtenido en la parte anterior?

3. **(Concavidad)** Obtenga la segunda derivada de la función $f$, nuevamente utilizando la regla del cuociente. Analice los signos de la expresión obtenida para obtener los intervalos en que $f''$ es positiva y negativa. Con esto, deduzca los intervalos de concavidad de $f$. Finalmente, con toda la información obtenida anteriormente, haga un esboce del gráfico de la función.

# Sesión 10: 8 de Septiembre

## La Regla de la Cadena

Con el objetivo de discutir el método de optimización de Newton, se debe hacer una pequeña discusión sobre polinomios de Taylor, en cuyo cálculo se debe emplear la regla de la cadena. Por lo que en esta sesión se discuten algunas consideraciones teóricas al respecto. En particular, se expone la versión equivalente de la derivada en un punto dada por

$$f'(x_0) = \lim_{x \to x_0} \frac{f(x) - f(x_0)}{x - x_0}$$

Y se discute la forma más teórica de la derivada dada por la siguiente definición alternativa

**Definición: (derivada)**
Si $f$ es función definida en una vecindad de $x_0 \in \R$, la derivada $f'(x_0) \in \R$ es aquel número real tal que $f$ admite una expansión del tipo

$$f(x) = f(x_0) + f'(x_0)(x - x_0) + \phi(x)$$

En que $\phi$ es una función que cumple que

$$\lim_{x \to x_0} \frac{\phi(x)}{x - x_0} = 0$$

**Nota:** El número $f'(x_0)$ discutido en la definición anterior a priori no tiene por qué existir, pero cuando existe, decimos que $f$ es derivable en $x_0$. Y se puede probar (pero no lo hicimos) que cuando $f'(x_0)$ existe, es único. En otras palabras, no existen dos ni más números que califiquen simultáneamente para ser la derivada de $f$ en $x_0$.

Con esta consideración teórica establecida, se enuncia la regla de la cadena:

**Teorema: (regla de la cadena)** Sea $f$ derivable en $x_0$ y $g$ derivable en $y_0 = f(x_0)$. Entonces, la composición $(g \circ f)(x) = g(f(x))$ es una función derivable en $x_0$ y su derivada viene dada por
$$(g \circ f)'(x_0) = g'(f(x_0)) \cdot f'(x_0)$$

# Sesión 11: 11 de Septiembre

## Demostración de la Regla de la Cadena

Se demostró la regla de la cadena utilizando la definición teórica de la derivada dada en la sesión anterior. Además, se inició la discusión sobre los polinomios de Taylor y sobre las preguntas que nos podríamos hacer sobre ellos.

Si bien se habló sobre las cotas para el error de aproximación, luego se decidió no discutirlas en más profundidad, ya que estas requieren el Teorema del Valor Medio, herramienta que no tenemos.

# Sesión 12: 17 de Septiembre

## Polinomios de Taylor

Nuestra aproximación consistió en polinomios del tipo

$$p_n(x) = \sum_{k = 0}^n a_k(x - x_0)^k$$

Y el criterio que define a un polinomio de Taylor es que el valor y las derivadas de orden hasta $n$ calcen. Es decir

$$\forall l \in \{0, \dots, n\} : p_n^{(l)}(x_0) = f^{(l)}(x_0)$$

Estas $n+1$ ecuaciones determinan, al menos en teoría, los $n+1$ coeficientes del polinomio.

Se utilizó lo anteriormente discutido sobre la regla de la cadena para probar la fórmula de taylor que otorga los coeficientes de forma explícita

$$a_l = \frac{f^{(l)}(x_0)}{l!}$$

Esto se implementó utilizando `sympy` luego de un tutorial introductorio en el notebook 1.

# Sesión 13: 22 de Septiembre

Utilizando lo previamente discutido, se expone y se programa en el notebook 2 el método de descenso. Queda pendiente programar el método de Newton.