from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///yappcd.sqlite'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


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


@app.route('/api/data')
def data():
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
