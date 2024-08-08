import React, { useState } from 'react';
import { Container, Row, Col, Form, Button, Image } from 'react-bootstrap';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';
import './Home.css';

const Home = () => {
  const [image, setImage] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [ethnicity, setEthnicity] = useState('');
  const [diabetes, setDiabetes] = useState('');
  const [cholesterolHDL, setCholesterolHDL] = useState('');
  const [MMSE, setMMSE] = useState('');
  const [functionalAssessment, setFunctionalAssessment] = useState('');
  const [memoryComplaints, setMemoryComplaints] = useState('');
  const [behavioralProblems, setBehavioralProblems] = useState('');
  const [ADL, setADL] = useState('');
  const [result, setResult] = useState(null);

  const handleImageChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setImage(URL.createObjectURL(e.target.files[0]));
      setImageFile(e.target.files[0]);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    const formData = new FormData();
    formData.append('ethnicity', ethnicity);
    formData.append('diabetes', diabetes);
    formData.append('cholesterol_HDL', cholesterolHDL);
    formData.append('MMSE', MMSE);
    formData.append('functional_assessment', functionalAssessment);
    formData.append('memory_complaints', memoryComplaints);
    formData.append('behavioral_problems', behavioralProblems);
    formData.append('ADL', ADL);
    formData.append('mri_image', imageFile);

    axios.post('http://localhost:5000/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
    .then(response => {
      setResult(response.data);
    })
    .catch(error => {
      console.error('There was an error!', error);
    });
  };

  return (
    <div className="home">
      <Container>
        <Row className="hero">
          <Col className='my-3'>
            <h1>Welcome to Alzheimer's Detection</h1>
            <p>Early detection of Alzheimer's using AI and MRI scans.</p>
          </Col>
        </Row>
        <Row className="form-section">
          <Col md={6}>
            <Form onSubmit={handleSubmit}>
              <Form.Group className="mb-3" controlId="ethnicity">
                <Form.Label>Enter Ethnicity</Form.Label>
                <Form.Control 
                  as="select" 
                  value={ethnicity} 
                  onChange={(e) => setEthnicity(e.target.value)}
                >
                  <option value="">Select Ethnicity</option>
                  <option value="0">Caucasian</option>
                  <option value="1">African American</option>
                  <option value="2">Asian</option>
                  <option value="3">Other</option>
                </Form.Control>
              </Form.Group>
              <Form.Group className="mb-3" controlId="diabetes">
                <Form.Label>Enter Diabetes</Form.Label>
                <Form.Control 
                  as="select" 
                  value={diabetes} 
                  onChange={(e) => setDiabetes(e.target.value)}
                >
                  <option value="">Select Diabetes</option>
                  <option value="0">No, I do not have Diabetes</option>
                  <option value="1">Yes, I do have Diabetes</option>
                </Form.Control>
              </Form.Group>
              <Form.Group className="mb-3" controlId="cholesterolHDL">
                <Form.Label>Enter Cholesterol HDL</Form.Label>
                <Form.Control 
                  as="textarea" 
                  rows={1} 
                  placeholder="Enter cholesterol HDL here" 
                  value={cholesterolHDL}
                  onChange={(e) => setCholesterolHDL(e.target.value)}
                />
              </Form.Group>
              <Form.Group className="mb-3" controlId="MMSE">
                <Form.Label>Enter MMSE</Form.Label>
                <Form.Control 
                  as="textarea" 
                  rows={1} 
                  placeholder="Enter MMSE here" 
                  value={MMSE}
                  onChange={(e) => setMMSE(e.target.value)}
                />
              </Form.Group>
              <Form.Group className="mb-3" controlId="functionalAssessment">
                <Form.Label>Enter Functional Assessment</Form.Label>
                <Form.Control 
                  as="textarea" 
                  rows={1} 
                  placeholder="Enter functional assessment here" 
                  value={functionalAssessment}
                  onChange={(e) => setFunctionalAssessment(e.target.value)}
                />
              </Form.Group>
              <Form.Group className="mb-3" controlId="memoryComplaints">
                <Form.Label>Enter Memory Complaints</Form.Label>
                <Form.Control 
                  as="textarea" 
                  rows={1} 
                  placeholder="Enter memory complaints here" 
                  value={memoryComplaints}
                  onChange={(e) => setMemoryComplaints(e.target.value)}
                />
              </Form.Group>
              <Form.Group className="mb-3" controlId="behavioralProblems">
                <Form.Label>Enter Behavioral Problems</Form.Label>
                <Form.Control 
                  as="textarea" 
                  rows={1} 
                  placeholder="Enter behavioral problems here" 
                  value={behavioralProblems}
                  onChange={(e) => setBehavioralProblems(e.target.value)}
                />
              </Form.Group>
              <Form.Group className="mb-3" controlId="ADL">
                <Form.Label>Enter ADL</Form.Label>
                <Form.Control 
                  as="textarea" 
                  rows={1} 
                  placeholder="Enter ADL here" 
                  value={ADL}
                  onChange={(e) => setADL(e.target.value)}
                />
              </Form.Group>
              <Form.Group className="mb-4" controlId="mriUpload">
                <Form.Label>Upload MRI Scan</Form.Label>
                <Form.Control type="file" accept="image/*" onChange={handleImageChange} />
              </Form.Group>
              <Button className="mb-4" variant="outline-light" type="submit">
                Get Results
              </Button>
            </Form>
          </Col>
          <Col md={6} className='img-col'>
            <div className="img-wrapper">
              {image ? <Image className='img-place' src={image} alt="Uploaded MRI" fluid /> : <div className="placeholder">Your image will appear here</div>}
            </div>
          </Col>
        </Row>
        {result && (
          <Row className="result-section">
            <Col>
              <h3>Prediction Result</h3>
              <p>Prediction: {result.prediction}</p>
              <p>Confidence: {result.confidence}</p>
            </Col>
          </Row>
        )}
      </Container>
    </div>
  );
};

export default Home;
