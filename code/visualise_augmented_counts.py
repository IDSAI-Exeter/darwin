import pandas
import matplotlib.pyplot as plt
import seaborn as sns


df = pandas.read_csv('../data/experiments/sample_species/augmented_counts.csv')

plot = []

for index, row in df.iterrows():
    plot.append({'species': row['name'], 'dataset': 'train', 'count': row['train']})
    plot.append({'species': row['name'], 'dataset': 'aug', 'count': row['aug']})


df = pandas.DataFrame(plot)

df = df.sort_values('count')
sns.barplot(orient='h',
            x='count',
            y='species',
            hue='dataset',
            data=df)
plt.tight_layout()
plt.show()