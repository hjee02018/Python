from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///yappcd.sqlite'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class yappann(db.Model):
    __tablename__ = 'yappann'
    pdbid = db.Column(db.Text, primary_key = True)
    prot_ch = db.Column(db.Text, primary_key = True)
    pep_ch = db.Column(db.Text, primary_key = True)
    pep_len = db.Column(db.Integer)
    rel_date = db.Column(db.Text)
    pep_seq = db.Column(db.Text)
    desc = db.Column(db.Text)

# table 생성
db.create_all()


# 라우팅 설정
@app.route('/')
def index():
    entries = yappann.query
    return render_template('basic_table.html', title='Basic Table',
                           entries=entries)


if __name__ == '__main__':
    app.run()