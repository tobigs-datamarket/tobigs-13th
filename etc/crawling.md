---
description: Python을 이용한 네이버 영화 크롤링
---

# Crawling

##  과제 내용 설명

* 대상: 예매순 상위 5개의 현재 상영 중인 영화
* 수집할 항목: 영화 제목, 주연배우 3인, 네티즌 평점, 관람객 평점, 기자/평론가 평점, 관람객 별점 리뷰 20건 공감순으로\(평점, 작성자닉네임, 리뷰본문\)

### 우수과제로 뽑은 이유

모듈화가 가장 잘 구현된 코드였습니다. 깔끔한 코딩으로 저장 형식이 과제 출제자의 의도 대로여서 좋았습니다. 다만, 데이터를 저장할 때 공백 문자를 제거\(strip\(\)\)하는 등의 개선할 부분도 보였습니다.

In \[1\]:

```python
import requests
from bs4 import BeautifulSoup
import re
```

#### 1. 예매순 상위 5개의 현재 상영 중인 영화 가져오기

영화 검색을 위한 5개의 영화 코드 리스트를 저장한다.In \[2\]:

```python
def get_movie_codes():
    codes = []                        #코드를 저장할 리스트
    
    url = 'https://movie.naver.com/movie/running/current.nhn'
    res = requests.get(url)
    html = res.text
    soup = BeautifulSoup(html, 'html.parser')
    
    i = 0
    for tag in soup.find('ul', class_='lst_detail_t1').find_all('li'):
        codes.append(tag.find('a').get('href')[28:])
        i += 1
        if i == 5 :
            break
    
    return codes
```

In \[3\]:

```python
codes = get_movie_codes()
codes
```

Out\[3\]:

```text
['179181', '186821', '187321', '186613', '181925']
```

#### 2. 영화 제목 가져오기

In \[4\]:

```python
def get_title(code):
    url = "https://movie.naver.com/movie/bi/mi/basic.nhn?code=" + code
    res = requests.get(url)
    html = res.text
    soup = BeautifulSoup(html, 'html.parser')
    
    title = soup.find('h3', class_='h_movie').find('a').text
    return title
```

In \[5\]:

```python
get_title("186613")
```

Out\[5\]:

```text
'작은 아씨들'
```

#### 3. 출연진 3명 가져오기

In \[6\]:

```python
def get_actor(code):
    url = "https://movie.naver.com/movie/bi/mi/basic.nhn?code=" + code
    res = requests.get(url)
    html = res.text
    soup = BeautifulSoup(html, 'html.parser')
    
    people = soup.find("div", class_="people").find_all('a', class_='tx_people')
    actor = []
    for i in range(1,4) :
        actor.append(people[i].text)
    
    return actor
```

In \[7\]:

```python
get_actor("187321")
```

Out\[7\]:

```text
['조지 맥케이', '딘-찰스 채프먼', '콜린 퍼스']
```

#### 4. 평점 가져오기

In \[8\]:

```python
def get_grade(code):
    url = "https://movie.naver.com/movie/bi/mi/basic.nhn?code=" + code
    res = requests.get(url)
    html = res.text
    soup = BeautifulSoup(html, 'html.parser')
    
    grade = {"audience_grade" : "",
             "critic_grade" : "",
             "netizen_grade" : ""
            }
    
    grades = []
    i = 0
    for tx in soup.find_all('div', class_='star_score'):
        num = ""
        for em in tx.find_all('em'):
            num += em.text
        grades.append(num)
        i += 1
        if i == 3 :
            break
    grade["audience_grade"] = grades[0]
    grade["critic_grade"] = grades[1]
    grade["netizen_grade"] = grades[2]
    
    return grade
```

In \[9\]:

```python
get_grade("187321")
```

Out\[9\]:

```text
{'audience_grade': '9.40', 'critic_grade': '7.67', 'netizen_grade': '9.02'}
```

#### 5. 관람객 평점 공감순 20건 가져오기

In \[10\]:

```python
def get_reviews(code):
    reviews = []
    for i in range(1,3) :
        url = "https://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?code=" + code + \
              "&type=after&isActualPointWriteExecute=false&isMileageSubscriptionAlready=false&isMileageSubscriptionReject=false&page=" + str(i)
        res = requests.get(url)
        html = res.text
        soup = BeautifulSoup(html, 'html.parser')
        
        for review in soup.find('div', class_="score_result").find_all("li") :
            grade = review.find('em').text
            user_id = review.find('div', class_='score_reple').find('dl').find('span').text
            comment = review.find('div', class_='score_reple').find('p').text.strip()
            reviews.append({'grade' : grade, 'user_id' : user_id, 'comment' : comment})
    return reviews
```

In \[11\]:

```python
get_reviews("179181")
```

Out\[11\]:

```text
[{'grade': '10',
  'user_id': 'bohemian(mabu****)',
  'comment': '난 전도연의 화류계 캐릭터가 좋다. 무뢰한, 너는 내 운명, 카운트다운...그리고 지푸라기'},
 {'grade': '10',
  'user_id': '최정규(cjg4****)',
  'comment': '전도연 연기 진짜 오진다...와 이 영화에서 완전 섹시하게 나온다 역시 명불허전임...'},
 {'grade': '10',
  'user_id': '달다(fxko****)',
  'comment': '8명의 배우가 모두 주인공 같은 느낌.'},
 {'grade': '1',
  'user_id': '어쩌라고(dpfk****)',
  'comment': '아니 개봉당일날 9시 땡하고 부터 평점 쏟아지는게 말이 돼냐? 요즘 조조는 꼭두새벽부터 함? 백번양보해서 시사회때 봤다 쳐도 이렇게나 많이 봤다고? 죄다 똑같은 말투에? 음원이고 영화고 조작질 역겹다 진짜'},
 {'grade': '9',
  'user_id': '써니(tlag****)',
  'comment': '개존잼 역시 전도연이죠? 카리스마 미쳐벌여ㅠㅁㅠ'},
 {'grade': '10',
  'user_id': '까칠소녀(oper****)',
  'comment': '연출, 연기, 스토리 모두 대박...무조건 보세요.'},
 {'grade': '9',
  'user_id': 'haeunnnnn(0_80****)',
  'comment': '관람객\n\n\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t진짜 보고싶었던 영환데 드디어 봤습니당 기다린 보람이 있네용ㅋㅋㅋ 등장인물이 많았는데 영화 속에서 잘 풀어낸 것 같아요 강추합니당 !!'},
 {'grade': '9', 'user_id': 'hojo****', 'comment': '한국식 피칠갑을 한 타란티노 영화'},
 {'grade': '9',
  'user_id': 'Cjin(dlck****)',
  'comment': '관람객\n\n\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t연기오지고 스릴오지고'},
 {'grade': '10',
  'user_id': 'trau****',
  'comment': '스토리가 짱짱하네요~ 심리적인 긴장감을 잘 살린 영화인것 같네요~ 인기좀 끌듯...'},
 {'grade': '10',
  'user_id': '꽁끼(bamb****)',
  'comment': '관람객\n\n\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t연기 쩐다잉 ,,, 또 보고 싶음 ㅠ'},
 {'grade': '1',
  'user_id': 'osk1****',
  'comment': '방금 보고 왔는데 지금 심정이 지푸라기라도 잡고 싶은 심정이다'},
 {'grade': '10',
  'user_id': 'Linus(getu****)',
  'comment': '전도연을 위한, 전도연에 의한 영화! 데뷔작이라고는 믿을수 없는 연출력에놀랐다~'},
 {'grade': '10',
  'user_id': 'myd5q3ji7(i2g1****)',
  'comment': '이 캐스팅 너무 마음에 든다.영화보고나서도 할말이 많아지는 영화'},
 {'grade': '9',
  'user_id': '파리투나잇(lgxe****)',
  'comment': '솔직히 이 영화 돈주고 볼만합니다ㅎㅎ'},
 {'grade': '1',
  'user_id': 'DooGi(vxor****)',
  'comment': '영화 보는 내내 제발 한순간만이라도 재미있는 장면 나오길 지푸라기 잡는 심정으로 봤는데 없음'},
 {'grade': '10',
  'user_id': '시무룩(simu****)',
  'comment': '관람객\n\n\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t전도연 등장하자마자 걍 스크린 장악함ㅋㅋㅋㅋ역시 전도연이 선택한 작품은 안보고 넘어갈 수 없지'},
 {'grade': '9',
  'user_id': '히피아(whdt****)',
  'comment': '다들너무연기를잘하고일단 이런 스토리탄탄한영화 오랜만이네요 굿굿구성도재밋고'},
 {'grade': '7',
  'user_id': '할렘(upge****)',
  'comment': '관람객\n\n\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t갖은 재료 다 집어넣었는데 왜 맛이 안나지?'},
 {'grade': '10', 'user_id': 'nu(zkkb****)', 'comment': '인정할 수 밖에 없는 영화'}]
```

#### 6. 영화별 내용 저장

In \[12\]:

```python
def dict_movie_contents(code):
    movie = {'title' : "", 
             'actor' : [], 
             'grade' : {},
             'reviews' : []
            }
    
    # 영화 제목, 출연진 3명, 평점 가져오기
    movie['title'] = get_title(code)
    movie['actor'] = get_actor(code)
    movie['grade'] = get_grade(code)
    
    # 리뷰 20건 가져오기
    movie['reviews'] = get_reviews(code)
    
    return movie
```

In \[13\]:

```python
dict_movie_contents("179181")
```

Out\[13\]:

```text
{'title': '지푸라기라도 잡고 싶은 짐승들',
 'actor': ['전도연', '정우성', '배성우'],
 'grade': {'audience_grade': '8.67',
  'critic_grade': '6.71',
  'netizen_grade': '7.17'},
 'reviews': [{'grade': '10',
   'user_id': 'bohemian(mabu****)',
   'comment': '난 전도연의 화류계 캐릭터가 좋다. 무뢰한, 너는 내 운명, 카운트다운...그리고 지푸라기'},
  {'grade': '10',
   'user_id': '최정규(cjg4****)',
   'comment': '전도연 연기 진짜 오진다...와 이 영화에서 완전 섹시하게 나온다 역시 명불허전임...'},
  {'grade': '10',
   'user_id': '달다(fxko****)',
   'comment': '8명의 배우가 모두 주인공 같은 느낌.'},
  {'grade': '1',
   'user_id': '어쩌라고(dpfk****)',
   'comment': '아니 개봉당일날 9시 땡하고 부터 평점 쏟아지는게 말이 돼냐? 요즘 조조는 꼭두새벽부터 함? 백번양보해서 시사회때 봤다 쳐도 이렇게나 많이 봤다고? 죄다 똑같은 말투에? 음원이고 영화고 조작질 역겹다 진짜'},
  {'grade': '9',
   'user_id': '써니(tlag****)',
   'comment': '개존잼 역시 전도연이죠? 카리스마 미쳐벌여ㅠㅁㅠ'},
  {'grade': '10',
   'user_id': '까칠소녀(oper****)',
   'comment': '연출, 연기, 스토리 모두 대박...무조건 보세요.'},
  {'grade': '9',
   'user_id': 'haeunnnnn(0_80****)',
   'comment': '관람객\n\n\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t진짜 보고싶었던 영환데 드디어 봤습니당 기다린 보람이 있네용ㅋㅋㅋ 등장인물이 많았는데 영화 속에서 잘 풀어낸 것 같아요 강추합니당 !!'},
  {'grade': '9', 'user_id': 'hojo****', 'comment': '한국식 피칠갑을 한 타란티노 영화'},
  {'grade': '9',
   'user_id': 'Cjin(dlck****)',
   'comment': '관람객\n\n\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t연기오지고 스릴오지고'},
  {'grade': '10',
   'user_id': 'trau****',
   'comment': '스토리가 짱짱하네요~ 심리적인 긴장감을 잘 살린 영화인것 같네요~ 인기좀 끌듯...'},
  {'grade': '10',
   'user_id': '꽁끼(bamb****)',
   'comment': '관람객\n\n\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t연기 쩐다잉 ,,, 또 보고 싶음 ㅠ'},
  {'grade': '1',
   'user_id': 'osk1****',
   'comment': '방금 보고 왔는데 지금 심정이 지푸라기라도 잡고 싶은 심정이다'},
  {'grade': '10',
   'user_id': 'Linus(getu****)',
   'comment': '전도연을 위한, 전도연에 의한 영화! 데뷔작이라고는 믿을수 없는 연출력에놀랐다~'},
  {'grade': '10',
   'user_id': 'myd5q3ji7(i2g1****)',
   'comment': '이 캐스팅 너무 마음에 든다.영화보고나서도 할말이 많아지는 영화'},
  {'grade': '9',
   'user_id': '파리투나잇(lgxe****)',
   'comment': '솔직히 이 영화 돈주고 볼만합니다ㅎㅎ'},
  {'grade': '1',
   'user_id': 'DooGi(vxor****)',
   'comment': '영화 보는 내내 제발 한순간만이라도 재미있는 장면 나오길 지푸라기 잡는 심정으로 봤는데 없음'},
  {'grade': '10',
   'user_id': '시무룩(simu****)',
   'comment': '관람객\n\n\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t전도연 등장하자마자 걍 스크린 장악함ㅋㅋㅋㅋ역시 전도연이 선택한 작품은 안보고 넘어갈 수 없지'},
  {'grade': '9',
   'user_id': '히피아(whdt****)',
   'comment': '다들너무연기를잘하고일단 이런 스토리탄탄한영화 오랜만이네요 굿굿구성도재밋고'},
  {'grade': '7',
   'user_id': '할렘(upge****)',
   'comment': '관람객\n\n\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t갖은 재료 다 집어넣었는데 왜 맛이 안나지?'},
  {'grade': '10', 'user_id': 'nu(zkkb****)', 'comment': '인정할 수 밖에 없는 영화'}]}
```

#### 7. 파일 저장하기

In \[21\]:

```python
#기본 타입은 py (json, txt 등 가능)
def save(file_name = "movies", save_type = "py"):
    # 1차 저장 (list 형태)
    movies = []
     
    # 상위 5개 영화 코드 불러오기
    codes = get_movie_codes()
    
    for code in codes :
        movies.append(dict_movie_contents(code))
    
    file = file_name + "." + save_type
    f = open(file, 'w', encoding='utf-8')
    f.write(str(movies))
    f.close()
```

In \[22\]:

```python
save()
```

