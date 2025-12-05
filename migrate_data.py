
from app import db, User, Sample, Experiment, Prefix, SampleTrash, ExperimentTrash
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

def migrate_to_postgres():
    """Migrate data from local SQLite database to Replit PostgreSQL."""
    load_dotenv()
    
    # Source database (SQLite)
    source_engine = create_engine('sqlite:///instance/project_tracker.db')
    SourceSession = sessionmaker(bind=source_engine)
    source_session = SourceSession()
    
    # Target database (Replit PostgreSQL)
    target_uri = os.getenv('DATABASE_URL')
    if not target_uri:
        print("Error: DATABASE_URL environment variable not set")
        return
    
    target_engine = create_engine(target_uri)
    TargetSession = sessionmaker(bind=target_engine)
    target_session = TargetSession()
    
    # Create all tables in PostgreSQL
    db.metadata.create_all(target_engine)
    
    # Migrate Users
    users = source_session.query(User).all()
    for user in users:
        target_session.add(User(
            id=user.id,
            username=user.username,
            password=user.password,
            is_admin=user.is_admin
        ))
    
    # Migrate Samples
    samples = source_session.query(Sample).all()
    for sample in samples:
        target_session.add(Sample(
            id=sample.id,
            date=sample.date,
            time=sample.time,
            am_pm=sample.am_pm,
            recipe_front=sample.recipe_front,
            recipe_back=sample.recipe_back,
            glass_type=sample.glass_type,
            length=sample.length,
            thickness=sample.thickness,
            height=sample.height,
            cleaning=sample.cleaning,
            coating=sample.coating,
            annealing=sample.annealing,
            done=sample.done
        ))
    
    # Migrate Experiments
    experiments = source_session.query(Experiment).all()
    for experiment in experiments:
        target_session.add(Experiment(
            id=experiment.id,
            transmittance=experiment.transmittance,
            reflectance=experiment.reflectance,
            absorbance=experiment.absorbance,
            plqy=experiment.plqy,
            sem=experiment.sem,
            edx=experiment.edx,
            xrd=experiment.xrd
        ))
    
    # Migrate Prefixes
    prefixes = source_session.query(Prefix).all()
    for prefix in prefixes:
        target_session.add(Prefix(
            prefix=prefix.prefix,
            full_form=prefix.full_form
        ))
    
    # Migrate Trash Data
    sample_trash = source_session.query(SampleTrash).all()
    for trash in sample_trash:
        target_session.add(SampleTrash(
            trash_id=trash.trash_id,
            id=trash.id,
            date=trash.date,
            time=trash.time,
            am_pm=trash.am_pm,
            recipe_front=trash.recipe_front,
            recipe_back=trash.recipe_back,
            glass_type=trash.glass_type,
            length=trash.length,
            thickness=trash.thickness,
            height=trash.height,
            cleaning=trash.cleaning,
            coating=trash.coating,
            annealing=trash.annealing,
            done=trash.done,
            deleted_at=trash.deleted_at,
            deleted_by=trash.deleted_by
        ))
    
    experiment_trash = source_session.query(ExperimentTrash).all()
    for trash in experiment_trash:
        target_session.add(ExperimentTrash(
            trash_id=trash.trash_id,
            id=trash.id,
            sample_trash_id=trash.sample_trash_id,
            transmittance=trash.transmittance,
            reflectance=trash.reflectance,
            absorbance=trash.absorbance,
            plqy=trash.plqy,
            sem=trash.sem,
            edx=trash.edx,
            xrd=trash.xrd,
            deleted_at=trash.deleted_at,
            deleted_by=trash.deleted_by
        ))
    
    # Commit changes
    target_session.commit()
    print("Migration completed successfully!")

if __name__ == "__main__":
    migrate_to_postgres()
