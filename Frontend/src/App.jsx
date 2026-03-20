import React, { useState } from 'react';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function App() {
  const [currentPage, setCurrentPage] = useState('home');

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif', backgroundColor: '#f8fafc', color: '#0f172a' }}>
      <Navbar setCurrentPage={setCurrentPage} />
      <main style={{ flex: 1, padding: '60px 0' }}>
        {currentPage === 'home' && <HomePage setCurrentPage={setCurrentPage} />}
        {currentPage === 'upload' && <UploadPage setCurrentPage={setCurrentPage} />}
        {currentPage === 'about' && <AboutPage setCurrentPage={setCurrentPage} />}
      </main>
      <Footer />
    </div>
  );
}

function Navbar({ setCurrentPage }) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <nav style={{ background: 'linear-gradient(135deg, #1e293b, #0f172a)', borderBottom: '1px solid rgba(255,255,255,0.1)', position: 'sticky', top: 0, zIndex: 100, boxShadow: '0 8px 32px rgba(0,0,0,0.3)' }}>
      <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '16px 24px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <button onClick={() => { setCurrentPage('home'); setMobileMenuOpen(false); }} style={{ display: 'flex', alignItems: 'center', gap: '12px', background: 'none', border: 'none', cursor: 'pointer', fontSize: '20px', fontWeight: '700', color: '#fff', padding: '8px 12px', borderRadius: '12px', transition: 'all 0.3s' }} onMouseEnter={(e) => e.target.style.backgroundColor = 'rgba(255,255,255,0.1)'} onMouseLeave={(e) => e.target.style.backgroundColor = 'transparent'}>
          <div style={{ width: '36px', height: '36px', background: 'linear-gradient(135deg, #da3bf6ff, #d360faff)', borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff', fontWeight: '700', fontSize: '14px', boxShadow: '0 4px 12px rgba(246, 59, 218, 0.4)' }}>BC</div>
          <span>BreastCare AI</span>
        </button>

        <ul style={{ display: 'flex', flexDirection: mobileMenuOpen ? 'column' : 'row', listStyle: 'none', gap: '12px', alignItems: 'center', position: mobileMenuOpen ? 'absolute' : 'relative', top: mobileMenuOpen ? '100%' : 'auto', left: mobileMenuOpen ? '0' : 'auto', right: mobileMenuOpen ? '0' : 'auto', background: mobileMenuOpen ? 'rgba(15,23,42,0.95)' : 'transparent', padding: mobileMenuOpen ? '16px 24px' : '0', width: mobileMenuOpen ? '100%' : 'auto', borderBottom: mobileMenuOpen ? '1px solid rgba(255,255,255,0.1)' : 'none' }}>
          {['Analysis', 'About'].map((item, idx) => (
            <li key={idx} style={{ borderBottom: mobileMenuOpen ? (idx < 1 ? '1px solid rgba(255,255,255,0.1)' : 'none') : 'none', paddingBottom: mobileMenuOpen ? '8px' : '0', width: mobileMenuOpen ? '100%' : 'auto' }}>
              <button onClick={() => { setCurrentPage(item === 'Analysis' ? 'upload' : item.toLowerCase()); setMobileMenuOpen(false); }} style={{ background: 'rgba(255,255,255,0.12)', border: '1px solid rgba(255,255,255,0.2)', cursor: 'pointer', padding: '10px 20px', fontSize: '15px', color: '#e2e8f0', borderRadius: '10px', transition: 'all 0.3s', fontWeight: '500' }} onMouseEnter={(e) => { e.target.style.backgroundColor = 'rgba(59,130,246,0.3)'; e.target.style.borderColor = 'rgba(59,130,246,0.5)'; e.target.style.color = '#fff'; }} onMouseLeave={(e) => { e.target.style.backgroundColor = 'rgba(255,255,255,0.12)'; e.target.style.borderColor = 'rgba(255,255,255,0.2)'; e.target.style.color = '#e2e8f0'; }}>
                {item}
              </button>
            </li>
          ))}
        </ul>
      </div>
    </nav>
  );
}

function HomePage({ setCurrentPage }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '80px', maxWidth: '1200px', margin: '0 auto', padding: '0 24px' }}>
      {/* Hero Section */}
      <section style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '60px', alignItems: 'center' }}>
        <div>
          <h1 style={{ fontSize: '48px', fontWeight: '800', marginBottom: '16px', lineHeight: '1.2' }}>Advanced Mammogram Analysis</h1>
          <p style={{ fontSize: '18px', color: '#475569', marginBottom: '32px', lineHeight: '1.6' }}>Leveraging artificial intelligence for accurate breast cancer detection and diagnosis support</p>
          <button onClick={() => setCurrentPage('upload')} style={{ padding: '16px 32px', background: '#0f172a', color: '#fff', border: 'none', borderRadius: '12px', fontSize: '16px', fontWeight: '600', cursor: 'pointer', transition: 'all 0.3s' }} onMouseEnter={(e) => { e.target.style.backgroundColor = '#0f172a'; e.target.style.boxShadow = '0 8px 16px rgba(37,99,235,0.3)'; e.target.style.transform = 'translateY(-2px)'; }} onMouseLeave={(e) => { e.target.style.backgroundColor = '#0f172a'; e.target.style.boxShadow = 'none'; e.target.style.transform = 'translateY(0)'; }}>Analyze Mammogram</button>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '400px', background: 'linear-gradient(135deg, rgba(37,99,235,0.1), rgba(59,130,246,0.1))', borderRadius: '20px', border: '2px solid rgba(37,99,235,0.2)', overflow: 'hidden' }}>
          <img src="/breast-cancer.jpg" alt="Breast Cancer Detection" style={{ width: '100%', height: '100%', objectFit: 'cover', borderRadius: '18px' }} />
        </div>
      </section>

      {/* Disclaimer */}
      <section style={{ background: '#fef3c7', borderLeft: '5px solid #f59e0b', padding: '24px', borderRadius: '12px' }}>
        <h3 style={{ color: '#92400e', marginBottom: '12px', fontSize: '18px' }}>Medical Disclaimer</h3>
        <p style={{ color: '#78350f', lineHeight: '1.6' }}>This application is intended for <strong>research and educational purposes only</strong>. It is not FDA-approved and should not be used as a substitute for professional medical evaluation. All results must be reviewed by a qualified radiologist or medical professional for clinical decision-making.</p>
      </section>

      {/* Features */}
      <section>
        <div style={{ textAlign: 'center', marginBottom: '48px' }}>
          <h2 style={{ fontSize: '36px', fontWeight: '800', marginBottom: '12px' }}>Key Features</h2>
          <p style={{ fontSize: '16px', color: '#475569' }}>Comprehensive analysis capabilities powered by state-of-the-art technology</p>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '24px' }}>
          {[
            { title: 'Advanced AI Analysis', desc: 'Utilizes Swin Transformer neural network specifically trained on mammography imaging data for precise detection' },
            { title: 'Rapid Processing', desc: 'Receive comprehensive analysis results in seconds with detailed confidence metrics and risk assessments' },
            { title: 'Detailed Reporting', desc: 'Generate professional-grade PDF reports with analysis results and visualizations for clinical review' },
            { title: 'Privacy Protection', desc: 'Uploaded images are processed securely and automatically deleted after analysis completion' }
          ].map((feature, idx) => (
            <div key={idx} style={{ background: '#fff', padding: '32px 24px', borderRadius: '16px', border: '1px solid #e2e8f0', textAlign: 'center', transition: 'all 0.3s', cursor: 'pointer' }} onMouseEnter={(e) => { e.currentTarget.style.borderColor = '#2563eb'; e.currentTarget.style.boxShadow = '0 10px 25px rgba(0,0,0,0.1)'; e.currentTarget.style.transform = 'translateY(-8px)'; }} onMouseLeave={(e) => { e.currentTarget.style.borderColor = '#e2e8f0'; e.currentTarget.style.boxShadow = 'none'; e.currentTarget.style.transform = 'translateY(0)'; }}>
              <h4 style={{ fontSize: '18px', marginBottom: '12px', color: '#0f172a' }}>{feature.title}</h4>
              <p style={{ color: '#475569', lineHeight: '1.6', fontSize: '14px' }}>{feature.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* How it Works */}
      <section>
        <div style={{ textAlign: 'center', marginBottom: '48px' }}>
          <h2 style={{ fontSize: '36px', fontWeight: '800', marginBottom: '12px' }}>How It Works</h2>
          <p style={{ fontSize: '16px', color: '#475569' }}>Simple four-step process for complete mammogram analysis</p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '20px', flexWrap: 'wrap', justifyContent: 'center' }}>
          {[
            { num: '1', title: 'Upload Images', desc: 'Submit both CC and MLO views of the mammogram in standard image formats' },
            { num: '2', title: 'AI Processing', desc: 'Advanced neural network analyzes images for suspicious regions and patterns' },
            { num: '3', title: 'Instant Results', desc: 'Receive prediction results with confidence scores and risk level assessment' },
            { num: '4', title: 'Generate Report', desc: 'Export professional PDF report for clinical review and documentation' }
          ].map((step, idx) => (
            <div key={idx} style={{ flex: '1', minWidth: '200px', background: '#fff', padding: '24px', borderRadius: '16px', border: '1px solid #e2e8f0', textAlign: 'center', transition: 'all 0.3s', cursor: 'pointer' }} onMouseEnter={(e) => { e.currentTarget.style.borderColor = '#2563eb'; e.currentTarget.style.boxShadow = '0 10px 25px rgba(0,0,0,0.1)'; e.currentTarget.style.transform = 'translateY(-4px)'; }} onMouseLeave={(e) => { e.currentTarget.style.borderColor = '#e2e8f0'; e.currentTarget.style.boxShadow = 'none'; e.currentTarget.style.transform = 'translateY(0)'; }}>
              <div style={{ display: 'inline-flex', alignItems: 'center', justifyContent: 'center', width: '48px', height: '48px', background: 'linear-gradient(135deg, #0f172a, #0f172a)', color: '#fff', borderRadius: '50%', fontWeight: '700', fontSize: '20px', marginBottom: '16px' }}>{step.num}</div>
              <h4 style={{ fontSize: '16px', marginBottom: '8px', color: '#0f172a' }}>{step.title}</h4>
              <p style={{ fontSize: '14px', color: '#475569', lineHeight: '1.6' }}>{step.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section style={{ background: 'linear-gradient(135deg, #0f172a, #0f172a)', color: '#fff', padding: '60px 40px', borderRadius: '20px', textAlign: 'center' }}>
        <h2 style={{ fontSize: '36px', marginBottom: '12px' }}>Ready to Get Started?</h2>
        <p style={{ fontSize: '18px', marginBottom: '32px', opacity: 0.95 }}>Upload your mammogram images now for comprehensive AI-powered analysis</p>
        <button onClick={() => setCurrentPage('upload')} style={{ padding: '16px 32px', background: '#fff', color: '#0f172a', border: 'none', borderRadius: '12px', fontSize: '16px', fontWeight: '600', cursor: 'pointer', transition: 'all 0.3s' }} onMouseEnter={(e) => { e.target.style.transform = 'translateY(-2px)'; e.target.style.boxShadow = '0 8px 16px rgba(0,0,0,0.2)'; }} onMouseLeave={(e) => { e.target.style.transform = 'translateY(0)'; e.target.style.boxShadow = 'none'; }}>Analyze Mammogram</button>
      </section>
    </div>
  );
}

function UploadPage({ setCurrentPage }) {
  const [ccImage, setCCImage] = useState(null);
  const [mloImage, setMLOImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleImageSelect = (e, setImage) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setImage({ file, preview: event.target.result });
        setError(null);
      };
      reader.readAsDataURL(file);
    } else {
      setError('Please select a valid image file');
    }
  };

  const handleDragDrop = (e, setImage) => {
    e.preventDefault();
    e.stopPropagation();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setImage({ file, preview: event.target.result });
        setError(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleGenerateReport = async () => {
    if (!ccImage || !mloImage) {
      setError('Images required for report generation');
      return;
    }

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('cc_view', ccImage.file);
      formData.append('mlo_view', mloImage.file);

      const response = await axios.post(`${API_URL}/generate-report`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        responseType: 'blob'
      });

      // Create a blob URL and trigger download
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'breast_cancer_detection_report.pdf');
      document.body.appendChild(link);
      link.click();
      link.parentNode.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Report generation error:', err);
      setError('Error generating report. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handlePredict = async () => {
    if (!ccImage || !mloImage) {
      setError('Please upload both CC and MLO views before analysis');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', ccImage.file);

      const response = await axios.post(`${API_URL}/predict`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      console.log('API Response:', response.data);

      setResults({
        prediction: response.data.prediction || 'unknown',
        confidence: Math.round((response.data.confidence || 0) * 100),
        probabilities: response.data.probabilities || { benign: 0.5, malignant: 0.5 },
        processing_time: response.data.processing_time_seconds || 0
      });
    } catch (err) {
      console.error('Prediction error:', err);
      setError(err.response?.data?.detail || 'Error processing images. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  if (results) {
    const isHighRisk = results.prediction.toLowerCase() === 'malignant';
    const riskColor = isHighRisk ? '#dc2626' : '#16a34a';

    return (
      <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '40px 24px' }}>
        <div style={{ background: '#fff', borderRadius: '20px', padding: '40px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }}>
          <div style={{ background: '#fbdbfeff', borderLeft: '5px solid #0f172a', padding: '24px', borderRadius: '12px', marginBottom: '32px' }}>
            <h4 style={{ color: '#0c4a6e', marginBottom: '8px' }}>Important Notice</h4>
            <p style={{ color: '#082f49' }}>These results are for informational and research purposes only. Professional medical evaluation by a qualified radiologist is required for clinical diagnosis and treatment decisions.</p>
          </div>

          <div style={{ marginBottom: '32px' }}>
            <h2 style={{ fontSize: '24px', marginBottom: '20px', color: '#0f172a' }}>Analysis Results</h2>
            <div style={{ borderLeft: '6px solid ' + riskColor, padding: '28px', background: '#f8fafc', borderRadius: '12px' }}>
              <p style={{ fontSize: '14px', fontWeight: '600', color: '#475569', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Classification</p>
              <p style={{ fontSize: '32px', fontWeight: '800', color: riskColor, marginBottom: '16px' }}>{results.prediction.toUpperCase()}</p>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', paddingTop: '16px', borderTop: '1px solid #e2e8f0' }}>
                <div>
                  <p style={{ fontSize: '12px', fontWeight: '600', color: '#475569', textTransform: 'uppercase' }}>Confidence</p>
                  <p style={{ fontSize: '24px', fontWeight: '700', color: '#0f172a' }}>{(results.confidence / 100).toFixed(2)}</p>
                </div>
                <div>
                  <p style={{ fontSize: '12px', fontWeight: '600', color: '#475569', textTransform: 'uppercase' }}>Risk Level</p>
                  <p style={{ fontSize: '24px', fontWeight: '700', color: riskColor }}>{isHighRisk ? 'High Risk' : 'Low Risk'}</p>
                </div>
              </div>
            </div>
          </div>

          <div style={{ marginBottom: '32px' }}>
            <h3 style={{ fontSize: '20px', marginBottom: '20px', color: '#0f172a' }}>Probability Distribution</h3>
            {['benign', 'malignant'].map((type) => (
              <div key={type} style={{ marginBottom: '20px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', fontSize: '14px', fontWeight: '600', color: '#0f172a' }}>
                  <span>{type.charAt(0).toUpperCase() + type.slice(1)}</span>
                  <span style={{ color: '#0f172a', fontWeight: '700' }}>{Math.round(results.probabilities[type] * 100)}%</span>
                </div>
                <div style={{ height: '8px', background: '#e2e8f0', borderRadius: '4px', overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: Math.round(results.probabilities[type] * 100) + '%', background: type === 'benign' ? 'linear-gradient(90deg, #16a34a, #4ade80)' : 'linear-gradient(90deg, #dc2626, #f87171)', transition: 'width 0.8s ease' }}></div>
                </div>
              </div>
            ))}
          </div>

          <div style={{ marginBottom: '32px', background: isHighRisk ? '#fee2e2' : '#dcfce7', borderLeft: '5px solid ' + (isHighRisk ? '#dc2626' : '#16a34a'), padding: '20px', borderRadius: '12px' }}>
            <h4 style={{ color: isHighRisk ? '#7f1d1d' : '#14532d', marginBottom: '8px', fontSize: '16px' }}>{isHighRisk ? 'High Risk Classification' : 'You Are Safe ✓'}</h4>
            <p style={{ color: isHighRisk ? '#991b1b' : '#166534', lineHeight: '1.6' }}>
              {isHighRisk 
                ? 'The analysis indicates suspicious findings. Immediate consultation with a qualified radiologist is strongly recommended.'
                : 'No suspicious findings detected. The mammogram analysis shows benign results. Continue with regular screening as recommended by medical professionals.'}
            </p>
          </div>

          <div style={{ display: 'flex', gap: '16px', justifyContent: 'center', flexWrap: 'wrap', marginTop: '40px' }}>
            <button onClick={handleGenerateReport} disabled={loading} style={{ padding: '16px 32px', background: 'linear-gradient(135deg, #0f172a, #0f172a)', color: '#fff', border: 'none', borderRadius: '12px', fontSize: '16px', fontWeight: '600', cursor: loading ? 'not-allowed' : 'pointer', transition: 'all 0.3s', opacity: loading ? 0.6 : 1 }} onMouseEnter={(e) => { if (!loading) { e.target.style.transform = 'translateY(-2px)'; e.target.style.boxShadow = '0 8px 16px rgba(218, 59, 246, 0.4)'; } }} onMouseLeave={(e) => { if (!loading) { e.target.style.transform = 'translateY(0)'; e.target.style.boxShadow = 'none'; } }}>
              {loading ? 'Generating...' : 'Generate Report'}
            </button>
            <button onClick={() => { setCCImage(null); setMLOImage(null); setResults(null); }} style={{ padding: '16px 32px', background: '#0f172a', color: '#fff', border: 'none', borderRadius: '12px', fontSize: '16px', fontWeight: '600', cursor: 'pointer', transition: 'all 0.3s' }} onMouseEnter={(e) => { e.target.style.backgroundColor = '#0f172a'; e.target.style.transform = 'translateY(-2px)'; }} onMouseLeave={(e) => { e.target.style.backgroundColor = '#0f172a'; e.target.style.transform = 'translateY(0)'; }}>Analyze Another</button>
            <button onClick={() => setCurrentPage('home')} style={{ padding: '16px 32px', background: 'transparent', color: '#0f172a', border: '2px solid #0f172a', borderRadius: '12px', fontSize: '16px', fontWeight: '600', cursor: 'pointer', transition: 'all 0.3s' }} onMouseEnter={(e) => { e.target.style.backgroundColor = 'rgba(37,99,235,0.08)'; e.target.style.transform = 'translateY(-2px)'; }} onMouseLeave={(e) => { e.target.style.backgroundColor = 'transparent'; e.target.style.transform = 'translateY(0)'; }}>Back to Home</button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '40px 24px' }}>
      <div style={{ textAlign: 'center', marginBottom: '48px' }}>
        <h1 style={{ fontSize: '36px', fontWeight: '800', marginBottom: '12px', color: '#0f172a' }}>Mammogram Analysis</h1>
        <p style={{ fontSize: '16px', color: '#475569' }}>Upload both CC (Craniocaudal) and MLO (Mediolateral Oblique) views for comprehensive analysis</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '32px', marginBottom: '40px' }}>
        {[{ label: 'CC View', sublabel: 'Craniocaudal', state: ccImage, setState: setCCImage }, { label: 'MLO View', sublabel: 'Mediolateral Oblique', state: mloImage, setState: setMLOImage }].map((item, idx) => (
          <div key={idx} style={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: '16px', padding: '32px', transition: 'all 0.3s' }}>
            <h3 style={{ fontSize: '18px', fontWeight: '600', color: '#0f172a', marginBottom: '4px' }}>{item.label}</h3>
            <p style={{ fontSize: '14px', color: '#475569', marginBottom: '20px' }}>{item.sublabel}</p>

            {item.state ? (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '12px' }}>
                <div style={{ width: '100%', height: '200px', borderRadius: '12px', overflow: 'hidden', background: '#f8fafc' }}>
                  <img src={item.state.preview} alt={item.label} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                </div>
                <div style={{ textAlign: 'center' }}>
                  <p style={{ fontSize: '14px', fontWeight: '600', color: '#0f172a', wordBreak: 'break-all' }}>{item.state.file.name}</p>
                  <p style={{ fontSize: '12px', color: '#475569' }}>{(item.state.file.size / 1024).toFixed(2)} KB</p>
                </div>
              </div>
            ) : (
              <div style={{ border: '2px dashed #e2e8f0', borderRadius: '12px', padding: '40px 20px', textAlign: 'center', cursor: 'pointer', transition: 'all 0.3s', background: '#f8fafc' }} onDragOver={(e) => { e.preventDefault(); e.currentTarget.style.borderColor = '#2563eb'; e.currentTarget.style.backgroundColor = 'rgba(37,99,235,0.05)'; }} onDragLeave={(e) => { e.currentTarget.style.borderColor = '#e2e8f0'; e.currentTarget.style.backgroundColor = '#f8fafc'; }} onDrop={(e) => { handleDragDrop(e, item.setState); e.currentTarget.style.borderColor = '#e2e8f0'; e.currentTarget.style.backgroundColor = '#f8fafc'; }}>
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '12px' }}>
                  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#475569" strokeWidth="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                    <polyline points="17 8 12 3 7 8"></polyline>
                    <line x1="12" y1="3" x2="12" y2="15"></line>
                  </svg>
                  <p style={{ fontSize: '14px', fontWeight: '600', color: '#0f172a' }}>Drag and drop your image here</p>
                  <p style={{ fontSize: '13px', color: '#475569', margin: '8px 0' }}>or</p>
                  <input type="file" accept="image/*" onChange={(e) => handleImageSelect(e, item.setState)} style={{ display: 'none' }} id={`input-${idx}`} />
                  <label htmlFor={`input-${idx}`} style={{ display: 'inline-block', padding: '10px 20px', background: '#0f172a', color: '#fff', borderRadius: '12px', cursor: 'pointer', fontSize: '14px', fontWeight: '600', transition: 'all 0.3s' }} onMouseEnter={(e) => { e.target.style.backgroundColor = '#0f172a'; e.target.style.transform = 'translateY(-2px)'; }} onMouseLeave={(e) => { e.target.style.backgroundColor = '#0f172a'; e.target.style.transform = 'translateY(0)'; }}>Select from Computer</label>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      {error && (
        <div style={{ background: '#fee2e2', borderLeft: '5px solid #dc2626', padding: '16px', borderRadius: '12px', color: '#991b1b', display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px', fontSize: '14px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: '24px', height: '24px', background: '#dc2626', color: '#fff', borderRadius: '50%', fontWeight: '700' }}>!</div>
          <span>{error}</span>
        </div>
      )}

      <div style={{ display: 'flex', gap: '16px', justifyContent: 'center', flexWrap: 'wrap', marginBottom: '40px' }}>
        <button onClick={handlePredict} disabled={loading || !ccImage || !mloImage} style={{ padding: '16px 32px', background: loading || !ccImage || !mloImage ? '#ccc' : '#0f172a', color: '#fff', border: 'none', borderRadius: '12px', fontSize: '16px', fontWeight: '600', cursor: loading || !ccImage || !mloImage ? 'not-allowed' : 'pointer', transition: 'all 0.3s', opacity: loading || !ccImage || !mloImage ? 0.6 : 1 }} onMouseEnter={(e) => { if (!loading && ccImage && mloImage) { e.target.style.backgroundColor = '#0f172a'; e.target.style.transform = 'translateY(-2px)'; } }} onMouseLeave={(e) => { if (!loading && ccImage && mloImage) { e.target.style.backgroundColor = '#0f172a'; e.target.style.transform = 'translateY(0)'; } }}>
          {loading ? 'Analyzing...' : 'Analyze Images'}
        </button>
        <button onClick={() => setCurrentPage('home')} style={{ padding: '16px 32px', background: 'transparent', color: '#0f172a', border: '2px solid #0f172a', borderRadius: '12px', fontSize: '16px', fontWeight: '600', cursor: 'pointer', transition: 'all 0.3s' }} onMouseEnter={(e) => { e.target.style.backgroundColor = 'rgba(37,99,235,0.08)'; e.target.style.transform = 'translateY(-2px)'; }} onMouseLeave={(e) => { e.target.style.backgroundColor = 'transparent'; e.target.style.transform = 'translateY(0)'; }}>Back to Home</button>
      </div>

      <div style={{ background: '#f8fafc', padding: '16px 24px', borderRadius: '12px', textAlign: 'center' }}>
        <h4 style={{ fontSize: '14px', fontWeight: '600', color: '#0f172a', marginBottom: '4px' }}>Supported Formats</h4>
        <p style={{ fontSize: '13px', color: '#475569' }}>JPG, PNG, BMP, TIFF - Maximum 50MB per image</p>
      </div>
    </div>
  );
}

function AboutPage({ setCurrentPage }) {
  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '40px 24px' }}>
      <div style={{ textAlign: 'center', marginBottom: '60px' }}>
        <h1 style={{ fontSize: '36px', fontWeight: '800', marginBottom: '12px', color: '#0f172a' }}>About This Application</h1>
        <p style={{ fontSize: '16px', color: '#475569' }}>Technical specifications and important information</p>
      </div>

      <div style={{ maxWidth: '900px', margin: '0 auto 48px' }}>
        {[
          {
            title: 'System Overview',
            content: 'BreastCare AI is a research and educational platform that leverages advanced deep learning technology to provide supportive analysis of mammographic images. This system is designed to demonstrate the potential of artificial intelligence in medical imaging while adhering to strict ethical and professional standards.'
          },
          {
            title: 'Model Architecture',
            content: 'The core of this application is based on Swin Transformer, a state-of-the-art vision transformer architecture that has demonstrated superior performance in medical image analysis tasks.',
            features: ['Vision Transformer with hierarchical windowed attention mechanisms', 'Trained on CBIS-DDSM mammography dataset', 'Input dimensions: 224 × 224 pixels', 'Output: Binary classification with confidence scores', 'Inference time: Less than 1 second per image']
          },
          {
            title: 'Training Dataset',
            content: 'The model was trained on the CBIS-DDSM (Curated Breast Imaging Suite - Digital Database for Screening Mammography) dataset, one of the largest and most comprehensive publicly available collections of mammographic images.'
          },
          {
            title: 'Important Disclaimers',
            disclaimers: [
              { label: 'Research Purpose Only', text: 'This application is provided exclusively for educational, research, and demonstration purposes.' },
              { label: 'Not FDA Approved', text: 'This system has not received FDA approval or equivalent regulatory clearance from any government health authority.' },
              { label: 'Not Medical Advice', text: 'The output of this system does not constitute medical diagnosis, medical advice, or clinical recommendation.' },
              { label: 'Professional Review Required', text: 'All results must be reviewed and validated by a board-certified radiologist or qualified medical professional.' },
              { label: 'Privacy and Data Protection', text: 'Uploaded images are processed using secure methods and are not stored permanently on our servers.' }
            ]
          }
        ].map((section, idx) => (
          <div key={idx} style={{ background: '#fff', padding: '32px', borderRadius: '16px', marginBottom: '24px', border: '1px solid #e2e8f0', backgroundColor: section.disclaimers ? '#fef3c7' : '#fff', borderColor: section.disclaimers ? '#f59e0b' : '#e2e8f0' }}>
            <h2 style={{ fontSize: '24px', fontWeight: '700', marginBottom: '16px', color: section.disclaimers ? '#92400e' : '#0f172a' }}>{section.title}</h2>
            {section.content && <p style={{ color: section.disclaimers ? '#78350f' : '#475569', lineHeight: '1.8', marginBottom: '16px' }}>{section.content}</p>}
            {section.features && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {section.features.map((feature, i) => (
                  <div key={i} style={{ padding: '12px 16px', background: '#f8fafc', borderLeft: '3px solid #2563eb', borderRadius: '4px', color: '#475569' }}>
                    {feature}
                  </div>
                ))}
              </div>
            )}
            {section.disclaimers && (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '20px' }}>
                {section.disclaimers.map((disc, i) => (
                  <div key={i} style={{ background: '#fff', padding: '20px', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
                    <h4 style={{ fontSize: '16px', fontWeight: '600', marginBottom: '10px', color: '#0f172a' }}>{disc.label}</h4>
                    <p style={{ fontSize: '14px', color: '#475569', lineHeight: '1.6' }}>{disc.text}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      <div style={{ textAlign: 'center' }}>
        <button onClick={() => setCurrentPage('home')} style={{ padding: '16px 32px', background: 'transparent', color: '#0f172a', border: '2px solid #0f172a', borderRadius: '12px', fontSize: '16px', fontWeight: '600', cursor: 'pointer', transition: 'all 0.3s' }} onMouseEnter={(e) => { e.target.style.backgroundColor = 'rgba(37,99,235,0.08)'; e.target.style.transform = 'translateY(-2px)'; }} onMouseLeave={(e) => { e.target.style.backgroundColor = 'transparent'; e.target.style.transform = 'translateY(0)'; }}>Back to Home</button>
      </div>
    </div>
  );
}

function Footer() {
  return (
    <footer style={{ background: '#0f172a', color: '#fff', marginTop: 'auto', borderTop: '1px solid rgba(255,255,255,0.1)' }}>
      <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '48px 24px' }}>
        <div style={{ textAlign: 'center', fontSize: '13px', color: 'rgba(255,255,255,0.6)', display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <p>&copy; 2026 BreastCare AI. All rights reserved.</p>
          <p>Always consult qualified medical professionals for diagnosis and treatment decisions.</p>
          <p>For research purposes only, not approved for clinical use.</p>
        </div>
      </div>
    </footer>
  );
}