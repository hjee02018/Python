from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy

# flask어플리켕션 객체 생성
# __name__ : 현재 모듈
app = Flask(__name__)

# SQLAlchemy 설정 지정(yappcd.sqlite사용)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///yappcd.sqlite'
# SQLAlchemy 모델 변경을 추적하는 기능 사용X (메모리 절약)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# SQLAlchemy 객체 생성 & 어플리케이션 연결
db = SQLAlchemy(app)

# SQLAlchemy를 사용해 Database 모델 정의
class yappann(db.Model):
    __tablename__ = 'yappann'
    pdbid = db.Column(db.Text, primary_key=True)
    prot_ch = db.Column(db.Text, primary_key=True)
    pep_ch = db.Column(db.Text, primary_key=True)
    pep_len = db.Column(db.Integer)
    rel_date = db.Column(db.Text)
    pep_seq = db.Column(db.Text)
    desc = db.Column(db.Text)

    def to_dict(self):
        return {
            'pdbid': self.pdbid,
            'prot_ch': self.prot_ch,
            'pep_ch': self.pep_ch,
            'pep_len': self.pep_len,
            'pep_seq': self.pep_seq,
            'desc': self.desc
        }


@app.route('/')
def index():
    return render_template('browseyapp_ss.html', title='Browse YAPPCD')


@app.route('/cv')
def index1():
    return render_template('complexview.html', title='Browse YAPPCD')


@app.route('/cdn')
def index2():
    return render_template('cdnexample.html', title='Browse YAPPCD')

# ajax요청 처리
@app.route('/api/data')
def data():
    # 'yappann' 모델로부터 데이터베이스 쿼리 객체 생성
    query = yappann.query

    # search filter
    search = request.args.get('search[value]')
    if search:
        query = query.filter(db.or_(
            yappann.pdbid.like(f'%{search}%'),
            yappann.pep_seq.like(f'%{search}%'),
            yappann.desc.like(f'%{search}%')
        ))
    total_filtered = query.count()

    # sorting
    order = []
    i = 0
    while True:
        col_index = request.args.get(f'order[{i}][column]')
        if col_index is None:
            break
        col_name = request.args.get(f'columns[{col_index}][data]')
        if col_name not in ['n', 'age', 'email']:
            col_name = 'pdbid'
        descending = request.args.get(f'order[{i}][dir]') == 'desc'
        col = getattr(yappann, col_name)
        if descending:
            col = col.desc()
        order.append(col)
        i += 1
    if order:
        query = query.order_by(*order)

    # pagination
    start = request.args.get('start', type=int)
    length = request.args.get('length', type=int)
    query = query.offset(start).limit(length)

    # response
    return {
        'data': [user.to_dict() for user in query],
        'recordsFiltered': total_filtered,
        'recordsTotal': yappann.query.count(),
        'draw': request.args.get('draw', type=int),
    }


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    # 외부 접속 허용
    app.run(host='0.0.0.0')
