from django.shortcuts import render
from django.http import HttpResponse
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
import base64
import pandas_datareader.data as d


# Create your views here.
def home(request):
    return render(request, 'visualization/index.html')


def plot(request):
    img_path = 'visualization/static/visualization/img/buffer.png'
    if request.method == 'POST':
        company = request.POST['company']
        end_date = request.POST['end_date']
        start_date = request.POST['start_date']
    code_dict = {'S&P 500': '^GSPC',
                 'Technology SPDR': 'XLK', 'Finance SPDR': 'XLF',
                 'Apple': 'AAPL', 'Google': 'GOOG', 'Goldman Sachs': 'GS', 'JP Morgan': 'JPM'}
    prices = d.DataReader(code_dict[company], 'yahoo', start_date, end_date)['Adj Close'].values
    plt.plot(prices)
    plt.savefig(img_path)
    plt.close()
    with open(img_path, "rb") as f:
        encoded_string = base64.b64encode(f.read())
    return HttpResponse(encoded_string, content_type="image/png")
